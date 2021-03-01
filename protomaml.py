import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
from maml import *


class ProtoMAMLFewShotClassifier(MAMLFewShotClassifier):
    def __init__(self, device, args):
        super(ProtoMAMLFewShotClassifier, self).__init__(device, args)

        # self.prototype_sim_lambda = torch.FloatTensor([args.prototype_sim_lambda]).to(device) # factor to multiply the similarity loss with
        self.do_centralize = args.protomaml_do_centralize

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

        # Unfreeze slow model weights
        if epoch >= self.num_freeze_epochs:
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

            total_task_loss = 0

            with torch.no_grad():  # don't backprop through prototype creation
                support_embeddings = self.classifier(
                    input_ids=x_support_set_task,
                    attention_mask=len_support_set_task,
                    num_step=0,
                    return_pooled=True,
                    params=fast_weights,
                )[0]
            # compute prototypes
            prototypes = self.compute_prototypes(support_embeddings, y_support_set_task)

            # compute weights for classification layer
            fc_weight = self.protomaml_fc_weights(prototypes)
            fc_bias = self.protomaml_fc_bias(prototypes)

            # set weights
            (
                fast_weights["classifier.out_proj.weight"],
                fast_weights["classifier.out_proj.bias"],
            ) = (
                fc_weight.to(self.device),
                fc_bias.to(self.device),
            )

            for num_step in range(num_steps):
                torch.cuda.empty_cache()
                if torch.cuda.device_count() > 1:
                    torch.cuda.synchronize()

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
                    if torch.cuda.device_count() > 1:
                        torch.cuda.synchronize()

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

                    task_lang_log.append(target_loss.detach().item())
                    task_lang_log.append(accuracy)

            # Achieve gradient accumulation by already backpropping current loss
            torch.cuda.empty_cache()
            task_losses = torch.sum(torch.stack(task_losses)) / meta_batch_size
            task_losses.backward()
            total_task_loss += task_losses.detach().cpu().item()
            losses["loss"] += total_task_loss

            task_lang_logs.append(task_lang_log)

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

        # if self.classifier is not None:
        # 	self.classifier.unfreeze()

        if train_on_cpu:
            self.device = torch.device("cpu")

        self.inner_loop_optimizer.requires_grad_(False)
        self.inner_loop_optimizer.eval()

        self.inner_loop_optimizer.to(self.device)

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
        x_support_set_task, mask, y_support_set_task = batch

        # Get prototypes and init class head weights
        with torch.no_grad():
            support_embeddings = self.classifier(
                input_ids=x_support_set_task,
                attention_mask=mask,
                num_step=0,
                return_pooled=True,
                params=fast_weights,
            )[0]

        # compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, y_support_set_task)

        # compute weights for classification layer
        fc_weight = self.protomaml_fc_weights(prototypes)
        fc_bias = self.protomaml_fc_bias(prototypes)

        # set weights
        (
            fast_weights["classifier.out_proj.weight"],
            fast_weights["classifier.out_proj.bias"],
        ) = (
            fc_weight.to(self.device),
            fc_bias.to(self.device),
        )

        del prototypes
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

                x, mask, y_true = batch

                #########################################################
                # Start of actual finetuning
                #########################################################

                for train_step in range(self.args.number_of_training_steps_per_iter):
                    torch.cuda.empty_cache()
                    # fast_model.set_fast_weights(names_weights_copy)
                    support_loss = self.net_forward(
                        x,
                        mask=mask,
                        teacher_unary=y_true,
                        num_step=train_step,
                        fast_model=fast_weights,
                        training=True,
                    )

                    fast_weights = self.apply_inner_loop_update(
                        loss=support_loss,
                        names_weights_copy=fast_weights,
                        use_second_order=False,
                        current_step_idx=train_step,
                    )

                    if writer is not None:  # create histogram of weights
                        for param_name, param in fast_weights.items():
                            writer.add_histogram(
                                task_name + "/" + param_name, param, train_step + 1
                            )
                        writer.flush()

                    pbar_train.update(1)
                    pbar_train.set_description(
                        "finetuning phase {} -> loss: {}".format(
                            batch_idx * self.args.number_of_training_steps_per_iter
                            + train_step
                            + 1,
                            support_loss.item(),
                        )
                    )

                #########################################################
                # Evaluate finetuned model
                #########################################################
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
                                fast_model=fast_weights,
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
