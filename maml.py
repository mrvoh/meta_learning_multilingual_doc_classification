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
from transformers import AdapterConfig, AdapterType

from few_shot_learning_system import *
from meta_bert import MetaBERT, distil_state_dict_to_bert
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from ranger import Ranger


class MAMLFewShotClassifier(FewShotClassifier):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__(device, args)

        config = AutoConfig.from_pretrained(args.pretrained_weights)
        config.num_labels = args.num_classes_per_set
        model_initialization = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_weights, config=config
        )

        slow_model = MetaBERT

        # Init fast model
        state_dict = model_initialization.state_dict()
        config = model_initialization.config

        del model_initialization

        # Slow model
        self.classifier = slow_model.init_from_pretrained(
            state_dict,
            config,
            num_labels=args.num_classes_per_set,
            is_distil=self.is_distil,
            is_xlm=self.is_xlm,
            per_step_layer_norm_weights=args.per_step_layer_norm_weights,
            num_inner_loop_steps=args.number_of_training_steps_per_iter,
            device=device,
        )
        self.classifier.to("cpu")
        self.classifier.train()

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=torch.device("cpu"),
            init_learning_rate=self.task_learning_rate,
            total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
            use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate,
            init_class_head_lr_multiplier=self.args.init_class_head_lr_multiplier,
        )

        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(
                params=self.classifier.named_parameters()
            )
        )

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = Ranger(
            [
                {"params": self.classifier.parameters(), "lr": args.meta_learning_rate},
                {
                    "params": self.inner_loop_optimizer.parameters(),
                    "lr": args.meta_inner_optimizer_learning_rate,
                },
            ],
            lr=args.meta_learning_rate,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs * self.args.total_iter_per_epoch,
            eta_min=self.args.min_learning_rate,
        )

        self.inner_loop_optimizer.to(self.device)

        self.clip_value = 1.0
        # gradient clipping
        for p in self.classifier.parameters():
            if p.requires_grad:
                p.register_hook(
                    lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value)
                )

        self.num_freeze_epochs = args.num_freeze_epochs
        if self.num_freeze_epochs > 0:
            self.classifier.freeze()

    def get_inner_loop_parameter_dict(self, params, adapter_only=False):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                key = (
                    name.replace("module.", "", 1)
                    if name.startswith("module.")
                    else name
                )
                if self.args.enable_inner_loop_optimizable_ln_params:
                    param_dict[key] = param.to(device=self.device)
                else:
                    if "LayerNorm" not in key:
                        if adapter_only:
                            if "adapter" in key or "classifier" in key:
                                param_dict[key] = param.to(device=self.device)
                            else:
                                print(key)
                        else:
                            param_dict[key] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(
        self,
        loss,
        names_weights_copy,
        use_second_order,
        current_step_idx,
        allow_unused=True,
    ):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """

        all_names = list(names_weights_copy.keys())
        names_weights_copy = {
            k: v for k, v in names_weights_copy.items() if v is not None
        }

        grads = torch.autograd.grad(
            loss,
            names_weights_copy.values(),
            create_graph=use_second_order,
            allow_unused=allow_unused,
        )

        names_grads_wrt_params = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_wrt_params,
            num_step=current_step_idx,
        )
        del names_grads_wrt_params

        for name in all_names:
            if name not in names_weights_copy.keys():
                names_weights_copy[name] = None

        return names_weights_copy

    def net_forward(
        self,
        x,
        teacher_unary,
        fast_model,
        training,
        num_step,
        return_nr_correct=False,
        mask=None,
        task_name="",
    ):
        student_logits = self.classifier(
            input_ids=x, attention_mask=mask, num_step=num_step, params=fast_model
        )[0]

        set_kl_loss = False
        if task_name in self.gold_label_tasks and self.meta_loss.lower() == "kl":
            set_kl_loss = True
            self.meta_loss = "ce"

        loss = self.inner_loss(
            student_logits, teacher_unary, return_nr_correct=return_nr_correct
        )

        if set_kl_loss:
            self.meta_loss = "kl"

        return loss

    def forward(
        self,
        data_batch,
        epoch,
        use_second_order,
        num_steps,
        training_phase,
    ):
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
        if self.num_freeze_epochs <= epoch:
            self.classifier.unfreeze()

        losses = {"loss": 0}
        task_accuracies = []
        task_lang_logs = []

        for (
            task_id,
            (
                x_support_set_task,
                len_support_set_task,
                y_support_set_task,
                x_target_set_task,
                len_target_set_task,
                y_target_set_task,
                teacher_name,
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

            task_lang_log = [teacher_name, epoch]
            task_losses = []

            # freeze and unfreeze if necessary to get correct params
            if epoch <= self.num_freeze_epochs:
                self.classifier.unfreeze()

            fast_weights = self.classifier.get_inner_loop_params()

            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()

            total_task_loss = 0

            x_support_set_task = x_support_set_task.squeeze()
            len_support_set_task = len_support_set_task.squeeze()
            y_support_set_task = y_support_set_task.squeeze()
            x_target_set_task = x_target_set_task.squeeze()
            len_target_set_task = len_target_set_task.squeeze()
            y_target_set_task = y_target_set_task.squeeze()

            for num_step in range(num_steps):
                torch.cuda.empty_cache()

                support_loss, is_correct = self.net_forward(
                    x=x_support_set_task,
                    mask=len_support_set_task,
                    num_step=num_step,
                    teacher_unary=y_support_set_task,
                    fast_model=fast_weights,
                    training=True,
                    return_nr_correct=True,
                    task_name=teacher_name,
                )

                fast_weights = self.apply_inner_loop_update(
                    loss=support_loss,
                    names_weights_copy=fast_weights,
                    use_second_order=use_second_order,
                    current_step_idx=num_step,
                )

                if num_step == (self.args.number_of_training_steps_per_iter - 1):
                    # store support set statistics
                    task_lang_log.append(support_loss.detach().item())
                    task_lang_log.append(np.mean(is_correct))

                    target_loss, is_correct = self.net_forward(
                        x=x_target_set_task,
                        mask=len_target_set_task,
                        teacher_unary=y_target_set_task,
                        num_step=num_step,
                        fast_model=fast_weights,
                        training=True,
                        return_nr_correct=True,
                        task_name=teacher_name,
                    )

                    task_losses.append(target_loss)
                    accuracy = np.mean(is_correct)
                    task_accuracies.append(accuracy)
                    # store query set statistics
                    task_lang_log.append(target_loss.detach().item())
                    task_lang_log.append(accuracy)

            # Achieve gradient accumulation by already backpropping current loss
            torch.cuda.empty_cache()
            task_losses = torch.sum(torch.stack(task_losses)) / meta_batch_size

            task_losses.backward()
            total_task_loss += task_losses.detach().cpu().item()
            losses["loss"] += total_task_loss

            task_lang_logs.append(task_lang_log)

            torch.cuda.synchronize()

        losses["accuracy"] = np.mean(task_accuracies)
        if training_phase:
            return losses, task_lang_logs
        else:
            return losses

    def finetune_epoch(
        self,
        names_weights_copy,
        model_config,
        train_dataloader,
        dev_dataloader,
        best_loss,
        eval_every,
        model_save_dir,
        task_name,
        epoch,
        train_on_cpu=False,
        writer=None,
    ):
        """
        Finetunes the meta-learned classifier on a dataset
        :param train_dataloader: Dataloader with train examples
        :param dev_dataloader: Dataloader with validation examples
        :param best_loss: best achieved loss on dev set up till now
        :param eval_every: eval on dev set after eval_every updates
        :param model_save_dir: directory to save the model to
        :param task_name: name of the task finetuning is performed on
        :param epoch: current epoch number
        :return: best_loss
        """
        if train_on_cpu:
            self.device = torch.device("cpu")

        self.inner_loop_optimizer.requires_grad_(False)
        self.inner_loop_optimizer.eval()

        self.inner_loop_optimizer.to(self.device)
        self.classifier.to(self.device)

        if names_weights_copy is None:
            if epoch <= self.num_freeze_epochs:
                self.classifier.unfreeze()
            # # Get fast weights
            names_weights_copy = self.classifier.get_inner_loop_params()

            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()
        eval_every = (
            eval_every if eval_every < len(train_dataloader) else len(train_dataloader)
        )

        if writer is not None:  # create histogram of weights
            for param_name, param in names_weights_copy.items():
                writer.add_histogram(task_name + "/" + param_name, param, 0)
            writer.flush()

        with tqdm(
            initial=0, total=eval_every * self.args.number_of_training_steps_per_iter
        ) as pbar_train:

            for batch_idx, batch in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                batch = tuple(t.to(self.device) for t in batch)
                x, mask, y_true = batch

                for train_step in range(self.args.number_of_training_steps_per_iter):

                    support_loss = self.net_forward(
                        x,
                        mask=mask,
                        teacher_unary=y_true,
                        num_step=train_step,
                        fast_model=names_weights_copy,
                        training=True,
                    )

                    names_weights_copy = self.apply_inner_loop_update(
                        loss=support_loss,
                        names_weights_copy=names_weights_copy,
                        use_second_order=False,
                        current_step_idx=train_step,
                    )

                    self.inner_loop_optimizer.zero_grad()

                    pbar_train.update(1)
                    pbar_train.set_description(
                        "finetuning phase {} -> loss: {}".format(
                            batch_idx * self.args.number_of_training_steps_per_iter
                            + train_step
                            + 1,
                            support_loss.item(),
                        )
                    )

                    if writer is not None:  # create histogram of weights
                        for param_name, param in names_weights_copy.items():
                            writer.add_histogram(
                                task_name + "/" + param_name, param, train_step + 1
                            )
                        writer.flush()

                if (batch_idx + 1) % eval_every == 0:
                    print("Evaluating model...")
                    losses = []
                    is_correct_preds = []

                    if train_on_cpu:
                        self.device = torch.device("cuda")
                        self.classifier.to(self.device)

                    with torch.no_grad():
                        for batch in tqdm(
                            dev_dataloader,
                            desc="Evaluating",
                            leave=False,
                            total=len(dev_dataloader),
                        ):
                            batch = tuple(t.to(self.device) for t in batch)
                            x, mask, y_true = batch

                            loss, is_correct = self.net_forward(
                                x,
                                mask=mask,
                                teacher_unary=y_true,
                                fast_model=names_weights_copy,
                                training=False,
                                return_nr_correct=True,
                                num_step=train_step,
                            )
                            losses.append(loss.item())
                            is_correct_preds.extend(is_correct.tolist())

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
                        torch.save(
                            names_weights_copy,
                            os.path.join(
                                model_save_dir,
                                "model_finetuned_{}".format(
                                    task_name.replace("train/", "", 1)
                                    .replace("val/", "", 1)
                                    .replace("test/", "", 1)
                                ),
                            ),
                        )
                    return names_weights_copy, best_loss, avg_loss, accuracy
