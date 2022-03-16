from dataloader import *
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import os
import contextlib
from contextlib import contextmanager
import platform
from protomaml import (
    ProtoMAMLFewShotClassifier,
    TRIPLET_lOSS_KEY,
    CE_LOSS_KEY,
    TOTAL_LOSS_KEY,
    CONSISTENCY_LOSS_KEY,
    INTERPOLATION_LOSS_KEY,
)

from copy import deepcopy
import random
import torch
import torch.nn.functional as F


def pairwise_distances(x, y=None):
    """
        source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist)
    return torch.clamp(dist, 0.0, np.inf)


class ThreeWayProtoMAMLFewShotClassifier(ProtoMAMLFewShotClassifier):
    def __init__(self, device, args):
        super(ThreeWayProtoMAMLFewShotClassifier, self).__init__(device, args)

        self.use_swa = args.use_swa
        self.num_swa = args.num_swa
        self.use_majority_vote = args.use_majority_vote
        self.num_majority_votes = args.num_majority_votes

    def get_most_similar_classes(self, prototypes):

        distances = pairwise_distances(prototypes)
        n_classes = distances.size(0)

        # add arbitrarily large distance to self
        distances += torch.diag(torch.tensor([10**6] * n_classes).to(self.device))

        _, min_dist_ix = torch.min(distances.view(-1), dim=0)

        c1 = min_dist_ix // n_classes
        c2 = min_dist_ix % n_classes

        return c1, c2

    def construct_triplets1(
        self,
        fast_weights,
        x_support_adaptation,
        len_support_adaptation,
        x_support_prototype,
        len_support_prototype,
        class_indices,
    ):
        samples_per_class = (
            self.args.num_samples_per_class // 2
        )  # number is doubled to account for adaptation and prototype set

        c1, c2 = class_indices
        c1, c2 = int(c1.detach().cpu().numpy()), int(c2.detach().cpu().numpy())

        sample_index = list(
            range(c1 * samples_per_class, (c1 * samples_per_class) + samples_per_class)
        ) + list(
            range(c2 * samples_per_class, (c2 * samples_per_class) + samples_per_class)
        )
        # Filter support and adaptation set to only contain "hard" classes
        x_support_adaptation = x_support_adaptation[sample_index]
        len_support_adaptation = len_support_adaptation[sample_index]
        x_support_prototype = x_support_prototype[sample_index]
        len_support_prototype = len_support_prototype[sample_index]
        # Stack
        x = torch.cat((x_support_adaptation, x_support_prototype))
        mask = torch.cat((len_support_adaptation, len_support_prototype))
        # Get embeddings
        logits, embeddings = self.classifier(
            input_ids=x,
            attention_mask=mask,
            num_step=self.args.number_of_training_steps_per_iter - 1,
            return_pooled=True,
            params=fast_weights,
        )

        # create triplets
        c1_index = list(range(samples_per_class)) + list(
            range(2 * samples_per_class, 3 * samples_per_class)
        )
        c2_index = list(range(samples_per_class, 2 * samples_per_class)) + list(
            range(3 * samples_per_class, 4 * samples_per_class)
        )

        c1_embeddings = embeddings[c1_index].split(split_size=1)
        c2_embeddings = embeddings[c2_index].split(split_size=1)

        triplets = []
        # C1 as anchor class
        for ix, anchor in enumerate(c1_embeddings, start=1):
            for pos in c1_embeddings[ix:]:
                for neg in c2_embeddings:
                    triplets.append((anchor, pos, neg))

        # C2 as anchor class
        for ix, anchor in enumerate(c2_embeddings, start=1):
            for pos in c2_embeddings[ix:]:
                for neg in c1_embeddings:
                    triplets.append((anchor, pos, neg))

        return triplets

    def split_support_set(self, x_support_set, len_support_set, y_support_set):
        """
        Splits the support set into a prototype and adaptation set
        """
        # Split out samples for prototype creation and fast adaptation
        x_support_set_task = x_support_set.squeeze().split(
            split_size=1
        )  # int(self.args.num_samples_per_class / 2))
        x_support_prototype = torch.cat(x_support_set_task[0:][::2])
        x_support_adaptation = torch.cat(x_support_set_task[1:][::2])

        assert x_support_adaptation.size(0) == x_support_prototype.size(
            0
        ), "Size mismatch between adaptation and prototype set"
        len_support_set_task = len_support_set.squeeze().split(split_size=1)
        len_support_prototype = torch.cat(len_support_set_task[0:][::2])
        len_support_adaptation = torch.cat(len_support_set_task[1:][::2])
        assert len_support_adaptation.size(0) == len_support_prototype.size(
            0
        ), "Size mismatch between adaptation and prototype set"

        y_support_set_task = y_support_set.squeeze().split(split_size=1)
        y_support_prototype = torch.cat(y_support_set_task[0:][::2])
        y_support_adaptation = torch.cat(y_support_set_task[1:][::2])
        assert y_support_adaptation.size(0) == y_support_prototype.size(
            0
        ), "Size mismatch between adaptation and prototype set"

        return (
            x_support_prototype,
            len_support_prototype,
            y_support_prototype,
            x_support_adaptation,
            len_support_adaptation,
            y_support_adaptation,
        )

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
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """

        x_support_set = data_batch[SUPPORT_SET_SAMPLES_KEY]
        len_support_set = data_batch[SUPPORT_SET_LENS_KEY]
        x_target_set = data_batch[TARGET_SET_SAMPLES_KEY]
        len_target_set = data_batch[TARGET_SET_LENS_KEY]
        y_support_set = data_batch[SUPPORT_SET_ENCODINGS_KEY]
        y_target_set = data_batch[TARGET_SET_ENCODINGS_KEY]
        teacher_names = data_batch[SELECTED_CLASS_KEY]

        class_descr_x = data_batch.get(CLASS_NAMES_KEY, None)
        class_descr_len = data_batch.get(CLASS_NAMES_LENS_KEY, None)
        class_descr_y = data_batch.get(CLASS_NAMES_ENCODING_KEY, None)

        meta_batch_size = self.args.batch_size
        self.classifier.zero_grad()
        self.inner_loop_optimizer.zero_grad()
        self.classifier.train()

        # Unfreeze slow model weights
        if epoch >= self.num_freeze_epochs:
            self.classifier.unfreeze()

        losses = {TOTAL_LOSS_KEY: 0, CE_LOSS_KEY: 0}
        if self.use_triplet_loss:
            losses[TRIPLET_lOSS_KEY] = 0

        if self.use_consistency_loss:
            losses[CONSISTENCY_LOSS_KEY] = 0

        if self.use_convex_feature_space_loss:
            losses[INTERPOLATION_LOSS_KEY] = 0

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

            if epoch < self.num_freeze_epochs:
                self.classifier.unfreeze()

            # Get inner-loop parameters
            fast_weights = self.classifier.get_inner_loop_params()

            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()

            # Split out samples for prototype creation and fast adaptation
            (
                x_support_prototype,
                len_support_prototype,
                y_support_prototype,
                x_support_adaptation,
                len_support_adaptation,
                y_support_adaptation,
            ) = self.split_support_set(
                x_support_set_task, len_support_set_task, y_support_set_task
            )

            x_target_set_task = x_target_set_task.squeeze()
            len_target_set_task = len_target_set_task.squeeze()
            y_target_set_task = y_target_set_task.squeeze()

            # Split query set and augmented samples in case of consistency training
            (
                x_target_set_task,
                len_target_set_task,
                y_target_set_task,
                x_aug_set_task,
                len_aug_set_task,
            ) = self.split_query_and_aug(
                x_target_set_task=x_target_set_task,
                len_target_set_task=len_target_set_task,
                y_target_set_task=y_target_set_task,
                training_phase=training_phase,
            )

            # Get prototypes and fc weights
            prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
                fast_weights=fast_weights,
                x=x_support_prototype
                if class_descr_x is None
                else class_descr_x[task_id],
                mask=len_support_prototype
                if class_descr_len is None
                else class_descr_len[task_id],
                y=y_support_prototype
                if class_descr_y is None
                else class_descr_y[task_id],
            )

            # set weights
            (
                fast_weights["classifier.out_proj.weight"],
                fast_weights["classifier.out_proj.bias"],
            ) = (
                fc_weight.to(self.device),
                fc_bias.to(self.device),
            )

            ##############################################
            # Inner-loop updates
            ##############################################

            for num_step in range(num_steps):

                fast_weights, support_losses, is_correct = self.inner_update_step(
                    x=x_support_adaptation,
                    mask=len_support_adaptation,
                    num_step=num_step,
                    y=y_support_adaptation,
                    fast_weights=fast_weights,
                    teacher_name=teacher_name,
                    use_second_order=use_second_order,
                )

            ##############################################
            # Outer-loop update
            ##############################################
            task_lang_log.append(support_losses[TOTAL_LOSS_KEY].detach().item())
            task_lang_log.append(np.mean(is_correct))
            if torch.cuda.device_count() > 1:
                torch.cuda.synchronize()

            none_context = contextmanager(lambda: iter([None]))()
            context_manager = (
                torch.no_grad()
                if (self.consistency_training and task_id % 2 == 0)
                else none_context
            )

            with context_manager:
                res = self.net_forward(
                    x=x_target_set_task,
                    mask=len_target_set_task,
                    teacher_unary=y_target_set_task,
                    num_step=num_step,
                    fast_model=fast_weights,
                    training=True,
                    return_nr_correct=True,
                    task_name=teacher_name,
                    consistency_training=self.use_consistency_loss and training_phase,
                    aug_x=x_aug_set_task,
                    aug_mask=len_aug_set_task,
                )

            #####################################################
            # Consistency loss
            #####################################################
            if self.consistency_training and task_id % 2 == 0:
                res["losses"] = self.apply_consistency_training(
                    fast_model=fast_weights,
                    logits=res["logits"],
                    aug_x=x_aug_set_task,
                    aug_mask=len_aug_set_task,
                    y_true=y_target_set_task,
                )

            target_losses = res["losses"]
            is_correct = res["is_correct"]

            # Log and compute grads
            target_loss = target_losses[TOTAL_LOSS_KEY]
            if self.use_uncertainty_task_weighting:
                log_task_std = self.log_task_stds[teacher_name]
                target_loss = (
                    1 / torch.exp(-1 * log_task_std) ** 2
                ) * target_loss + log_task_std

            target_loss = target_loss / meta_batch_size
            accuracy = np.mean(is_correct)
            task_accuracies.append(accuracy)
            # store query set statistics
            task_lang_log.append(target_loss.detach().item())
            task_lang_log.append(accuracy)

            # Achieve gradient accumulation by already backpropping current loss
            torch.cuda.empty_cache()
            target_loss.backward()

            # Log each individual loss
            for k in target_losses.keys():
                if k == TOTAL_LOSS_KEY:
                    losses[k] += (
                        target_losses[k].detach().cpu().item() / meta_batch_size
                    )
                elif k in losses.keys():
                    losses[k] += target_losses[k] / meta_batch_size

            task_lang_logs.append(task_lang_log)

        losses["accuracy"] = np.mean(task_accuracies)
        if training_phase:
            return losses, task_lang_logs
        else:
            return losses

    def stochastic_weight_averaging(
        self, x, mask, y, fast_weights, num_steps, num_swa, teacher_name
    ):
        """
        Perform stochastic weight averaging
        """
        num_samples, num_classes = y.shape
        samples_per_class = num_samples // num_classes

        for i in range(num_swa):

            # Randomly shuffle the samples within each class
            shuffled_indices = [
                random.sample(
                    [
                        (class_ix_start * samples_per_class) + sample_ix
                        for sample_ix in range(samples_per_class)
                    ],
                    samples_per_class,
                )
                for class_ix_start in range(num_classes)
            ]
            # flatten indices
            shuffled_indices = [
                ix for class_list in shuffled_indices for ix in class_list
            ]

            # Split out samples for prototype creation and fast adaptation
            (
                x_support_prototype,
                len_support_prototype,
                y_support_prototype,
                x_support_adaptation,
                len_support_adaptation,
                y_support_adaptation,
            ) = self.split_support_set(
                x[shuffled_indices], mask[shuffled_indices], y[shuffled_indices]
            )

            # Get prototypes and fc weights
            prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
                fast_weights=fast_weights,
                x=x_support_prototype,
                mask=len_support_prototype,
                y=y_support_prototype,
            )

            # Set weights
            (
                fast_weights["classifier.out_proj.weight"],
                fast_weights["classifier.out_proj.bias"],
            ) = (
                fc_weight.to(self.device),
                fc_bias.to(self.device),
            )

            for num_step in range(num_steps):

                fast_weights, support_loss, is_correct = self.inner_update_step(
                    x=x_support_adaptation,
                    mask=len_support_adaptation,
                    num_step=num_step,
                    y=y_support_adaptation,
                    fast_weights=fast_weights,
                    teacher_name=teacher_name,
                    use_second_order=False,
                )
            if i == 0:
                final_weights = {
                    k: v.to("cpu") / num_swa for k, v in fast_weights.items()
                }
            else:
                final_weights = {
                    k: v + (fast_weights[k].to("cpu") / num_swa)
                    for k, v in final_weights.items()
                }

        fast_weights = {k: v.to(self.device) for k, v in final_weights.items()}
        return fast_weights

    def get_majority_vote(
        self,
        x,
        mask,
        y,
        fast_weights,
        num_steps,
        num_votes,
        teacher_name,
        eval_dataloader,
    ):
        """
        Perform majority voting based on ensemble
        """
        num_samples, num_classes = y.shape
        samples_per_class = num_samples // num_classes

        all_preds, all_loss = [], []
        for i in range(num_votes):

            # Randomly shuffle the samples within each class
            shuffled_indices = [
                random.sample(
                    [
                        (class_ix_start * samples_per_class) + sample_ix
                        for sample_ix in range(samples_per_class)
                    ],
                    samples_per_class,
                )
                for class_ix_start in range(num_classes)
            ]
            # flatten indices
            shuffled_indices = [
                ix for class_list in shuffled_indices for ix in class_list
            ]

            # Split out samples for prototype creation and fast adaptation
            (
                x_support_prototype,
                len_support_prototype,
                y_support_prototype,
                x_support_adaptation,
                len_support_adaptation,
                y_support_adaptation,
            ) = self.split_support_set(
                x[shuffled_indices], mask[shuffled_indices], y[shuffled_indices]
            )

            # Get prototypes and fc weights
            prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
                fast_weights=fast_weights,
                x=x_support_prototype,
                mask=len_support_prototype,
                y=y_support_prototype,
            )

            # Set weights
            (
                fast_weights["classifier.out_proj.weight"],
                fast_weights["classifier.out_proj.bias"],
            ) = (
                fc_weight.to(self.device),
                fc_bias.to(self.device),
            )

            with tqdm(initial=0, total=num_steps) as pbar_train:
                for num_step in range(num_steps):
                    fast_weights, support_losses, is_correct = self.inner_update_step(
                        x=x_support_adaptation,
                        mask=len_support_adaptation,
                        num_step=num_step,
                        y=y_support_adaptation,
                        fast_weights=fast_weights,
                        teacher_name=teacher_name,
                        use_second_order=False,
                    )

                    pbar_train.update(1)
                    desc = f"finetuning voter nr {i} - phase {num_step + 1} -> "
                    for k in support_losses.keys():
                        if k == TOTAL_LOSS_KEY:
                            desc += f"{k}: {support_losses[k].item()} "
                        else:
                            desc += f"{k}: {support_losses[k]} "
                    pbar_train.set_description(desc)

            # Get predictions
            loss, is_correct_preds = self.eval_dataset(
                fast_weights=fast_weights, dataloader=eval_dataloader, to_gpu=False
            )
            all_preds.append(is_correct_preds)
            all_loss.append(loss)

        individual_accs = [np.round(np.mean(preds), 4) for preds in all_preds]
        print("Individual accuracy of each voter: {}".format(individual_accs))

        all_preds = (np.array(all_preds).mean(axis=0) > 0.5).astype(int)

        avg_loss = defaultdict(list)
        for voter_loss in all_loss:
            # Compute average loss per voter
            avg_voter_loss = {}
            for k in voter_loss[0].keys():
                avg_voter_loss[k] = np.mean([loss[k] for loss in voter_loss])
                avg_loss[k].append(avg_voter_loss[k])

        # Take the average over all voters
        avg_loss = {k: np.mean(v) for k, v in avg_loss.items()}

        return avg_loss, all_preds

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
        **kwargs,
    ):

        if train_on_cpu:
            self.device = torch.device("cpu")

        self.inner_loop_optimizer.requires_grad_(False)
        self.inner_loop_optimizer.eval()
        self.classifier.eval()

        self.inner_loop_optimizer.to(self.device)

        if names_weights_copy is None:
            if epoch <= self.num_freeze_epochs:
                self.classifier.unfreeze()
            # # Get fast weights
            fast_weights = self.classifier.get_inner_loop_params()
            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()
        else:
            fast_weights = names_weights_copy

        batch = next(iter(deepcopy(train_dataloader)))
        batch = tuple(t.to(self.device) for t in batch)
        x_support_set_task, len_support_set_task, y_support_set_task = batch

        if self.use_majority_vote:
            avg_loss, is_correct_preds = self.get_majority_vote(
                x=x_support_set_task,
                mask=len_support_set_task,
                y=y_support_set_task,
                fast_weights=fast_weights,
                num_steps=self.args.number_of_training_steps_per_iter,
                num_votes=self.num_majority_votes,
                teacher_name=task_name,
                eval_dataloader=dev_dataloader,
            )
        else:
            if self.use_swa:
                fast_weights = self.stochastic_weight_averaging(
                    x=x_support_set_task,
                    mask=len_support_set_task,
                    y=y_support_set_task,
                    fast_weights=fast_weights,
                    num_steps=self.args.number_of_training_steps_per_iter,
                    num_swa=self.num_swa,
                    teacher_name=task_name,
                )
                num_step = self.args.number_of_training_steps_per_iter - 1
            else:

                # Split out samples for prototype creation and fast adaptation
                (
                    x_support_prototype,
                    len_support_prototype,
                    y_support_prototype,
                    x_support_adaptation,
                    len_support_adaptation,
                    y_support_adaptation,
                ) = self.split_support_set(
                    x_support_set_task, len_support_set_task, y_support_set_task
                )

                class_descr_x = kwargs.get(CLASS_NAMES_KEY, None)
                class_descr_len = kwargs.get(CLASS_NAMES_LENS_KEY, None)
                class_descr_y = kwargs.get(CLASS_NAMES_ENCODING_KEY, None)

                # Get prototypes and fc weights
                prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
                    fast_weights=fast_weights,
                    x=x_support_prototype
                    if class_descr_x is None
                    else class_descr_x.to(self.device),
                    mask=len_support_prototype
                    if class_descr_len is None
                    else class_descr_len.to(self.device),
                    y=y_support_prototype
                    if class_descr_y is None
                    else class_descr_y.to(self.device),
                )

                # set weights
                (
                    fast_weights["classifier.out_proj.weight"],
                    fast_weights["classifier.out_proj.bias"],
                ) = (
                    fc_weight.to(self.device),
                    fc_bias.to(self.device),
                )

                eval_every = (
                    eval_every
                    if eval_every < len(train_dataloader)
                    else len(train_dataloader)
                )

                if writer is not None:  # create histogram of weights
                    for param_name, param in fast_weights.items():
                        writer.add_histogram(task_name + "/" + param_name, param, 0)
                    writer.flush()

                #########################################################
                # Start of actual finetuning
                #########################################################

                with tqdm(
                    initial=0,
                    total=eval_every * self.args.number_of_training_steps_per_iter,
                ) as pbar_train:

                    x, mask, y_true = (
                        x_support_adaptation,
                        len_support_adaptation,
                        y_support_adaptation,
                    )
                    for num_step in range(self.args.number_of_training_steps_per_iter):
                        (
                            fast_weights,
                            support_losses,
                            is_correct,
                        ) = self.inner_update_step(
                            x=x,
                            mask=mask,
                            num_step=num_step,
                            y=y_true,
                            fast_weights=fast_weights,
                            teacher_name=task_name,
                            use_second_order=False,
                        )

                        pbar_train.update(1)
                        desc = f"finetuning phase {num_step + 1} -> "
                        for k in support_losses.keys():
                            if k == TOTAL_LOSS_KEY:
                                desc += f"{k}: {support_losses[k].item()} "
                            else:
                                desc += f"{k}: {support_losses[k]} "
                        pbar_train.set_description(desc)
                        if writer is not None:  # create histogram of weights
                            for param_name, param in fast_weights.items():
                                writer.add_histogram(
                                    task_name + "/" + param_name, param, num_step + 1
                                )
                            writer.flush()
            #########################################################
            # Evaluate finetuned model
            #########################################################
            losses, is_correct_preds = self.eval_dataset(
                fast_weights=fast_weights,
                dataloader=dev_dataloader,
                to_gpu=train_on_cpu,
            )

            avg_loss = {}
            for k in losses[0].keys():
                avg_loss[k] = np.mean([loss[k] for loss in losses])

        # Set back
        self.inner_loop_optimizer.requires_grad_(True)

        accuracy = np.mean(is_correct_preds)
        print("Accuracy", accuracy)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("New best finetuned model with loss {:.05f}".format(best_loss))
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
