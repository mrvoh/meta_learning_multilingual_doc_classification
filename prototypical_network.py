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
from ranger import Ranger


def euclidean_dist(query, prototypes):
    # compute distance from every emb in query to all prototypes
    return ((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(
        2
    )  # [8,1,768] - [1,4,768]


class PrototypicalNetworkFewShotClassifier(FewShotClassifier):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(PrototypicalNetworkFewShotClassifier, self).__init__(device, args)

        # Init slow model
        config = AutoConfig.from_pretrained(args.pretrained_weights, num_labels=4)
        self.classifier = AutoModel.from_pretrained(
            args.pretrained_weights, config=config
        )

        self.classifier.to(self.device)

        # init optimizer
        self.optimizer = Ranger(self.trainable_parameters(), lr=args.meta_learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs * self.args.total_iter_per_epoch,
            eta_min=self.args.min_learning_rate,
        )

    def compute_prototypes(self, embeddings, labels):
        """
        Computes the prototype per class based on embeddings and labels
        :param embeddings:
        :param ohe_labels:
        :return:
        """
        _, labels = labels.max(dim=1)
        ohe_labels = torch.zeros(labels.size(0), labels.max() + 1).to(
            embeddings.device
        )  # batch size x nr labels
        ohe_labels.scatter_(1, labels.unsqueeze(1), 1)  # create one hot encoding

        embeddings = embeddings.unsqueeze(1)
        ohe_labels = ohe_labels.unsqueeze(2)

        class_sums = (ohe_labels * embeddings).sum(0)
        samples_per_class = ohe_labels.sum(0)

        support_mean = embeddings.mean()

        prototypes = (class_sums / samples_per_class) - support_mean

        # standardize prototypes to be unit vectors
        prototypes = torch.nn.functional.normalize(prototypes)

        return prototypes, support_mean

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

            support_embeddings = self.classifier(
                x_support_set_task, attention_mask=len_support_set_task
            )[0][:, 0, :]
            query_embeddings = self.classifier(
                x_target_set_task, attention_mask=len_target_set_task
            )[0][:, 0, :]
            # compute prototypes
            prototypes, support_mean = self.compute_prototypes(
                support_embeddings, y_support_set_task
            )
            # center around 0 and L2 normalize
            query_embeddings = torch.nn.functional.normalize(
                query_embeddings - support_mean
            )

            # compute distance to query set
            logits = -euclidean_dist(query_embeddings, prototypes)
            # compute loss

            loss, is_correct = self.inner_loss(
                logits, y_target_set_task, return_nr_correct=True
            )
            accuracy = np.mean(is_correct)

            task_accuracies.append(accuracy)
            loss = loss / meta_batch_size
            loss.backward()

            losses["loss"] += loss.detach().cpu().item()

        losses["accuracy"] = np.mean(task_accuracies)
        return losses, []

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
        writer=None,
    ):

        if train_on_cpu:
            self.device = torch.device("cpu")

        # # init model
        if model is None:
            model = self.classifier

        model.to(self.device)

        eval_every = (
            eval_every if eval_every < len(train_dataloader) else len(train_dataloader)
        )
        total_iters = -1

        ###############################################################################
        # Start finetuning
        ###############################################################################
        for batch_idx, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            x_support, mask, y_support = batch = tuple(t.to(self.device) for t in batch)

            support_embeddings = model(x_support, attention_mask=mask)[0][:, 0, :]
            prototypes, support_mean = self.compute_prototypes(
                support_embeddings, y_support
            )

            ###############################################################################
            # Evaluate finetuned model
            ###############################################################################

            if (batch_idx + 1) % eval_every == 0:
                print("Evaluating model...")
                losses = []
                is_correct_preds = []

                if train_on_cpu:  # set back to gpu
                    self.device = torch.device("cuda")
                    model.to(self.device)
                    prototypes = prototypes.to(self.device)

                with torch.no_grad():
                    for batch in tqdm(
                        dev_dataloader,
                        desc="Evaluating",
                        leave=False,
                        total=len(dev_dataloader),
                    ):
                        batch = tuple(t.to(self.device) for t in batch)
                        x, mask, y_true = batch

                        query_embeddings = model(x, attention_mask=mask)[0][:, 0, :]

                        # center around 0 and L2 normalize
                        query_embeddings = torch.nn.functional.normalize(
                            query_embeddings - support_mean
                        )

                        # compute distance to query set
                        logits = -euclidean_dist(query_embeddings, prototypes)
                        # compute loss

                        loss, is_correct = self.inner_loss(
                            logits, y_true, return_nr_correct=True
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
                        "New best finetuned model with loss {:.05f}".format(best_loss)
                    )
                    model.save_pretrained(
                        os.path.join(
                            model_save_dir,
                            "model_finetuned_{}".format(
                                task_name.replace("train/", "", 1)
                                .replace("val/", "", 1)
                                .replace("test/", "", 1)
                            ),
                        )
                    )

                return model, best_loss, avg_loss, accuracy
