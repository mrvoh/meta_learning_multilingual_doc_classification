import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import gc
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig

from few_shot_learning_system import *
from optimizers.radam import RAdam
from ranger import Ranger
from utils.torch_utils import cross_entropy_with_probs


class ReptileFewShotClassifier(FewShotClassifier):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(ReptileFewShotClassifier, self).__init__(device, args)

        # Init slow model
        config = AutoConfig.from_pretrained(args.pretrained_weights)
        config.num_labels = args.num_classes_per_set
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_weights, config=config
        )

        # init optimizer
        self.optimizer = Ranger(
            self.classifier.parameters(), lr=args.meta_learning_rate
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs * self.args.total_iter_per_epoch,
            eta_min=self.args.min_learning_rate,
        )

    def forward(self, data_batch, num_steps, training_phase, **kwargs):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        (
            x_support_set,
            len_support_set,
            x_target_set,
            len_target_set,
            y_support_set,
            y_target_set,
            teacher_names,
        ) = data_batch
        meta_batch_size = self.args.batch_size
        self.classifier.zero_grad()
        losses = {"loss": 0}
        task_accuracies = []
        sum_gradients = []
        total_task_loss = 0
        for (
            task_id,
            (
                x_support_set_task,
                len_support_set_task,
                y_support_set_task,
                x_target_set_task,
                len_target_set_task,
                y_target_set_task,
                task_name,
            ),
        ) in enumerate(
            zip(
                x_support_set,
                len_support_set,
                y_support_set,
                x_target_set,
                len_target_set,
                y_target_set,
                teacher_names,
            )
        ):

            set_kl_loss = False
            if task_name in self.gold_label_tasks and self.meta_loss.lower() == "kl":
                set_kl_loss = True
                self.meta_loss = "ce"

            fast_model = deepcopy(self.classifier)
            fast_model.zero_grad()

            # Get to right device:
            if torch.cuda.device_count() > 1:
                fast_model = MyDataParallel(fast_model)
            fast_model.to(self.device)  # Also wrap in DataParallel?

            inner_optimizer = RAdam(
                params=fast_model.parameters(),
                lr=self.args.init_inner_loop_learning_rate,
            )
            fast_model.train()

            task_losses = []

            for num_step in range(num_steps):
                torch.cuda.empty_cache()

                student_logits = fast_model(
                    x_support_set_task, attention_mask=len_support_set_task
                )[0]

                support_loss = self.inner_loss(
                    student_logits, y_support_set_task, return_nr_correct=False
                )

                support_loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()

            # List slow and fast weights
            fast_model.to(torch.device("cpu"))
            meta_weights = list(self.classifier.parameters())
            fast_weights = list(fast_model.parameters())

            # REPTILE meta-update
            for i, (meta_params, fast_params) in enumerate(
                zip(meta_weights, fast_weights)
            ):
                gradient = (
                    meta_params - fast_params
                ) / self.args.init_inner_loop_learning_rate
                if task_id == 0:
                    sum_gradients.append(gradient)
                else:
                    sum_gradients[i] += gradient

            fast_model.to(self.device)

            # Eval on query set
            fast_model.eval()
            with torch.no_grad():

                student_logits = fast_model(
                    x_target_set_task, attention_mask=len_target_set_task
                )[0]

                target_loss, is_correct = self.inner_loss(
                    student_logits, y_target_set_task, return_nr_correct=True
                )
                task_losses.append(target_loss)
                accuracy = np.mean(is_correct)

                task_accuracies.append(accuracy)

            fast_model.to(torch.device("cpu"))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
            total_task_loss += target_loss.detach().cpu().item()

        if set_kl_loss:  # set loss back to KL loss after gold label task is finished
            self.meta_loss = "kl"
        # if training:
        # Average gradient across tasks
        sum_gradients = [grad / float(self.args.batch_size) for grad in sum_gradients]

        # Assign gradient for original model, then using optimizer to update its weights
        for i, param in enumerate(self.classifier.parameters()):
            param.grad = sum_gradients[i]

        del sum_gradients
        gc.collect()

        # Collect statistics
        losses["loss"] += total_task_loss / meta_batch_size
        losses["accuracy"] = np.mean(task_accuracies)

        # return losses
        if training_phase:
            return losses, []
        else:
            return losses

    def finetune_epoch(
        self,
        model,
        model_config,
        train_dataloader,
        dev_dataloader,
        best_loss,
        eval_every,
        model_save_dir,
        task_name,
        epoch,
        train_on_cpu=False,
    ):

        if train_on_cpu:
            self.device = torch.device("cpu")

        # init model
        if model is None:
            model = self.classifier

        fast_model = deepcopy(model)
        fast_model.train()
        fast_model.to(self.device)

        # init optimizer
        inner_optimizer = RAdam(
            params=(
                v
                for k, v in fast_model.named_parameters()
                if "layernorm" not in k.lower()
            ),
            lr=self.args.init_inner_loop_learning_rate,
        )

        eval_every = (
            eval_every if eval_every < len(train_dataloader) else len(train_dataloader)
        )
        total_iters = -1

        ###############################################################################
        # Start finetuning
        ###############################################################################

        with tqdm(
            initial=0, total=eval_every * self.args.number_of_training_steps_per_iter
        ) as pbar_train:

            for batch_idx, batch in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                batch = tuple(t.to(self.device) for t in batch)
                x, mask, y_true = batch

                for train_step in range(self.args.number_of_training_steps_per_iter):
                    torch.cuda.empty_cache()

                    student_logits = fast_model(x, attention_mask=mask)[0]

                    support_loss = self.inner_loss(
                        student_logits, y_true, return_nr_correct=False
                    )

                    support_loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()

                    pbar_train.update(1)
                    pbar_train.set_description(
                        "finetuning phase {} -> loss: {}".format(
                            train_step + 1, support_loss.item()
                        )
                    )

                ###############################################################################
                # Evaluate finetuned model
                ###############################################################################

                if (batch_idx + 1) % eval_every == 0:
                    print("Evaluating model...")
                    losses = []
                    is_correct_preds = []

                    if train_on_cpu:
                        self.device = torch.device("cuda")
                        fast_model.to(self.device)

                    with torch.no_grad():
                        for batch in tqdm(
                            dev_dataloader,
                            desc="Evaluating",
                            leave=False,
                            total=len(dev_dataloader),
                        ):
                            batch = tuple(t.to(self.device) for t in batch)
                            x, mask, y_true = batch

                            student_logits = fast_model(x, attention_mask=mask)[0]

                            loss, is_correct = self.inner_loss(
                                student_logits, y_true, return_nr_correct=True
                            )
                            losses.append(loss.item())
                            is_correct_preds.extend(is_correct.tolist())

                    # Gather statistics
                    avg_loss = np.mean(losses)
                    accuracy = np.mean(is_correct_preds)
                    print("Accuracy", accuracy)
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        print(
                            "New best finetuned model with loss {:.05f}".format(
                                best_loss
                            )
                        )
                        fast_model.save_pretrained(
                            os.path.join(
                                model_save_dir,
                                "model_finetuned_{}".format(
                                    task_name.replace("train/", "", 1)
                                    .replace("val/", "", 1)
                                    .replace("test/", "", 1)
                                ),
                            )
                        )

                    return fast_model, best_loss, avg_loss, accuracy
