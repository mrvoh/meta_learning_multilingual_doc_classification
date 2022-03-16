from dataloader import *
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import os
import contextlib
from contextlib import contextmanager
from maml import (
    MAMLFewShotClassifier,
    TRIPLET_lOSS_KEY,
    CE_LOSS_KEY,
    TOTAL_LOSS_KEY,
    CONSISTENCY_LOSS_KEY,
    INTERPOLATION_LOSS_KEY,
)


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


class ProtoMAMLFewShotClassifier(MAMLFewShotClassifier):
    def __init__(self, device, args):
        super(ProtoMAMLFewShotClassifier, self).__init__(device, args)

        self.do_centralize = args.protomaml_do_centralize

    def get_most_similar_classes(self, prototypes):

        distances = pairwise_distances(prototypes)
        n_classes = distances.size(0)

        # add arbitrarily large distance to self
        distances += torch.diag(torch.tensor([10**6] * n_classes).to(self.device))

        _, min_dist_ix = torch.min(distances.view(-1), dim=0)

        c1 = min_dist_ix // n_classes
        c2 = min_dist_ix % n_classes

        return c1, c2

    def construct_query_triplets(self, embeddings, class_indices, samples_per_class):

        c1_start, c2_start = (
            t.detach().cpu().item() * samples_per_class for t in class_indices
        )

        # create triplets
        c1_index = list(range(c1_start, c1_start + samples_per_class))
        c2_index = list(range(c2_start, c2_start + samples_per_class))

        # Index the right embeddings
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

    def protomaml_fc_weights(self, prototypes):

        fc_weight = 2 * prototypes

        return nn.Parameter(fc_weight)

    def protomaml_fc_bias(self, prototypes):

        fc_bias = -torch.norm(prototypes, dim=1)

        return nn.Parameter(fc_bias)

    def compute_prototypes(self, embeddings, labels):
        """
        Computes the prototype per class based on embeddings and labels
        :param embeddings:
        :param ohe_labels:
        :return:
        """
        num_labels = labels.size(1)
        _, labels = labels.max(dim=1)
        ohe_labels = torch.zeros(labels.size(0), labels.max() + 1).to(
            embeddings.device
        )  # batch size x nr labels
        ohe_labels.scatter_(1, labels.unsqueeze(1), 1)  # create one hot encoding

        embeddings = embeddings.unsqueeze(1)
        ohe_labels = ohe_labels.unsqueeze(2)

        class_sums = (ohe_labels * embeddings).sum(0)
        samples_per_class = ohe_labels.sum(0)

        prototypes = class_sums / samples_per_class

        # standardize prototypes to be unit vectors
        if self.do_centralize:
            prototypes = torch.nn.functional.normalize(prototypes)

        assert num_labels == prototypes.size(
            0
        ), "There is a mismatch between the number of generated prototypes ({}) and inferred nr of classes ({}): \n {}".format(
            prototypes.size(0), num_labels, labels
        )

        return prototypes

    def get_prototype_similarity(self, prototypes):

        num_prototypes = prototypes.size(0)
        normalized = torch.nn.functional.normalize(prototypes)  # creates unit vectors

        similarity = normalized @ normalized.t()

        upper = torch.triu(similarity, diagonal=1)
        sim = torch.max(
            upper.sum() / num_prototypes, torch.FloatTensor([0]).to(self.device)
        )

        return sim * self.prototype_sim_lambda

    def get_prototypes_and_weights(self, fast_weights, x, mask, y):
        # don't backprop through prototype creation
        self.classifier.eval()
        with torch.no_grad():

            support_embeddings = self.classifier(
                input_ids=x,
                attention_mask=mask,
                num_step=0,
                return_pooled=True,
                params=fast_weights,
            )[1]
        self.classifier.train()
        # compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, y)

        # compute weights for classification layer
        fc_weight = self.protomaml_fc_weights(prototypes)
        fc_bias = self.protomaml_fc_bias(prototypes)

        return prototypes, fc_weight, fc_bias

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
            task_losses = []

            if epoch < self.num_freeze_epochs:
                self.classifier.unfreeze()

            # Get inner-loop parameters
            fast_weights = self.classifier.get_inner_loop_params()

            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()

            x_support_set_task = x_support_set_task.squeeze()
            len_support_set_task = len_support_set_task.squeeze()
            y_support_set_task = y_support_set_task.squeeze()
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

            # Construct prototypes and weight initialization of final linear layer
            prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
                fast_weights=fast_weights,
                x=x_support_set_task,
                mask=len_support_set_task,
                y=y_support_set_task,
            )

            # set weights
            (
                fast_weights["classifier.out_proj.weight"],
                fast_weights["classifier.out_proj.bias"],
            ) = (
                fc_weight.to(self.device),
                fc_bias.to(self.device),
            )

            for num_step in range(num_steps):
                fast_weights, support_losses, is_correct = self.inner_update_step(
                    x=x_support_set_task,
                    mask=len_support_set_task,
                    num_step=num_step,
                    y=y_support_set_task,
                    fast_weights=fast_weights,
                    teacher_name=teacher_name,
                    use_second_order=False,
                )

            ##############################################
            # Outer-loop update
            ##############################################
            task_lang_log.append(support_losses[TOTAL_LOSS_KEY].detach().item())
            task_lang_log.append(np.mean(is_correct))
            if torch.cuda.device_count() > 1:
                torch.cuda.synchronize()

            none_context = contextmanager(
                lambda: iter([None])
            )()  # contextlib.nullcontext() if py_ver > 6 else
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
        **kwargs,
    ):

        if train_on_cpu:
            self.device = torch.device("cpu")

        self.inner_loop_optimizer.eval()
        self.classifier.eval()

        self.inner_loop_optimizer.to(self.device)
        # Save some computation / memory
        self.inner_loop_optimizer.requires_grad_(False)

        if names_weights_copy is None:
            if epoch <= self.num_freeze_epochs:
                self.classifier.unfreeze()
            # Get fast weights
            fast_weights = self.classifier.get_inner_loop_params()
            if epoch < self.num_freeze_epochs:
                self.classifier.freeze()
        else:

            fast_weights = names_weights_copy

        batch = next(iter(deepcopy(train_dataloader)))
        batch = tuple(t.to(self.device) for t in batch)
        x_support_set_task, len_support_set_task, y_support_set_task = batch

        # Get prototypes and fc weights
        prototypes, fc_weight, fc_bias = self.get_prototypes_and_weights(
            fast_weights=fast_weights,
            x=x_support_set_task,
            mask=len_support_set_task,
            y=y_support_set_task,
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
            eval_every if eval_every < len(train_dataloader) else len(train_dataloader)
        )

        if writer is not None:  # create histogram of weights
            for param_name, param in fast_weights.items():
                writer.add_histogram(task_name + "/" + param_name, param, 0)
            writer.flush()

        with tqdm(
            initial=0, total=eval_every * self.args.number_of_training_steps_per_iter
        ) as pbar_train:

            for batch_idx, batch in enumerate(train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)

                x, len_support_set_task, y_true = batch

                ##############################################
                # Inner-loop updates
                ##############################################
                for num_step in range(self.args.number_of_training_steps_per_iter):
                    fast_weights, support_losses, is_correct = self.inner_update_step(
                        x=x,
                        mask=len_support_set_task,
                        num_step=num_step,
                        y=y_true,
                        fast_weights=fast_weights,
                        teacher_name=task_name,
                        use_second_order=False,
                    )

                    pbar_train.update(1)
                    desc = f"finetuning phase {batch_idx * self.args.number_of_training_steps_per_iter + num_step + 1} -> "
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
            fast_weights=fast_weights, dataloader=dev_dataloader, to_gpu=train_on_cpu
        )

        # Set back
        self.inner_loop_optimizer.requires_grad_(True)

        avg_loss = {}
        for k in losses[0].keys():
            avg_loss[k] = np.mean([loss[k] for loss in losses])

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
