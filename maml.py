from few_shot_learning_system import *
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
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.distances import CosineSimilarity
import contextlib


from meta_bert import MetaBERT, distil_state_dict_to_bert
from meta_adapter_bert import MetaAdapterBERT, META_ADAPTER_NAME
from inner_loop_optimizers import LSLRGradientDescentLearningRule
from optimizers.radam import RAdam
from ranger import Ranger

TRIPLET_lOSS_KEY = "triplet_loss"
CE_LOSS_KEY = "ce_loss"
CONSISTENCY_LOSS_KEY = "consistency_loss"
TOTAL_LOSS_KEY = "loss"
INTERPOLATION_LOSS_KEY = "interpolation_loss"


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

        if args.use_convex_feature_space_loss:
            config.output_hidden_states = True

        model_initialization = AutoModel.from_pretrained(
            args.pretrained_weights, config=config
        )

        self.bert_config = config

        self.use_adapter = args.use_adapter

        if self.use_adapter:
            adapter_config = AdapterConfig.load(
                "houlsby", reduction_factor=5, original_ln_after=False
            )
            config.adapter_down_sample_size = int(
                config.hidden_size / adapter_config.reduction_factor
            )
            # add a new adapter with the loaded config
            model_initialization.add_adapter(
                META_ADAPTER_NAME, AdapterType.text_task, config=adapter_config
            )
            # freeze all weights except for the adapter weights
            model_initialization.train_adapter([META_ADAPTER_NAME])

        slow_model = MetaAdapterBERT if self.use_adapter else MetaBERT

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
        if self.use_adapter:
            self.classifier.train_adapter()
            self.classifier.to(self.device)
        else:
            self.classifier.to("cpu")

        self.classifier.train()

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=torch.device("cpu"),
            init_learning_rate=self.task_learning_rate,
            total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
            use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate,
            init_class_head_lr_multiplier=self.args.init_class_head_lr_multiplier,
            use_adapter=self.use_adapter,
        )

        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(
                params=self.classifier.named_parameters(), adapter_only=self.use_adapter
            )
        )

        # Task weights
        self.use_uncertainty_task_weighting = args.use_uncertainty_task_weighting
        self.log_task_stds = nn.ParameterDict(
            {
                k: nn.Parameter(torch.zeros(1, device=self.device))
                for k in os.listdir(os.path.join(args.dataset_path, "train"))
            }
        )

        # Scale loss to nr of classes
        self.scale_losses = args.scale_losses

        # Triplet loss
        self.use_triplet_loss = args.use_triplet_loss
        self.triplet_loss_margin = args.triplet_loss_margin
        self.triplet_loss_lambda = args.triplet_loss_lambda
        self.triplet_loss_in_inner_loop = args.triplet_loss_in_inner_loop

        # Initialize loss
        self.use_cosine_distance = args.use_cosine_distance
        dist_fn = CosineSimilarity() if self.use_cosine_distance else None
        self.triplet_loss_new = TripletMarginLoss(
            margin=self.triplet_loss_margin, distance=dist_fn
        )
        # Consistency loss
        self.consistency_training = self.args.use_consistency_loss

        # Convex feature space loss
        self.use_convex_feature_space_loss = args.use_convex_feature_space_loss
        self.convex_feature_space_loss_in_inner_loop = (
            args.convex_feature_space_loss_in_inner_loop
        )
        self.convex_feature_space_loss_lambda = args.convex_feature_space_loss_lambda
        self.convex_feature_space_loss_nr_steps = (
            args.convex_feature_space_loss_nr_steps
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
                {
                    "params": self.log_task_stds.parameters(),
                    "lr": args.meta_learning_rate,
                },
                {"params": self.classifier.parameters(), "lr": args.meta_learning_rate},
                {
                    "params": self.inner_loop_optimizer.parameters(),
                    "lr": args.meta_inner_optimizer_learning_rate,
                },
            ],
            lr=args.meta_learning_rate,
            num_epochs=args.total_epochs,
            num_batches_per_epoch=args.total_iter_per_epoch,
            weight_decay=0,
            warmdown_min_lr=args.min_learning_rate,
        )

        self.inner_loop_optimizer.to(self.device)

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
                if self.args.enable_inner_loop_optimizable_bn_params:
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

    def construct_triplets(self, embeddings, y, normalize=True):
        n_samples, n_classes = y.size()
        samples_per_class = n_samples // n_classes

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings)

        # split into samples per class
        embeddings = embeddings.split(split_size=1)

        embeddings_per_class = [
            embeddings[i * samples_per_class : (i + 1) * samples_per_class]
            for i in range(n_classes)
        ]
        triplets = []
        # Construct all possible combinations of triplets
        for class_ix, class_embedding in enumerate(embeddings_per_class):

            remaining_classes = [
                class_embedding
                for ix, class_embedding in enumerate(embeddings_per_class)
                if ix != class_ix
            ]
            # All remaining classes are negatives
            negatives = [
                emb for neg_class_emb in remaining_classes for emb in neg_class_emb
            ]
            for sample_ix, anchor in enumerate(class_embedding):
                # all remaining samples in same class are positives
                for pos in class_embedding[sample_ix + 1 :]:

                    for neg in negatives:
                        triplets.append((anchor, pos, neg))

        return triplets

    def get_mix_layer(self):

        return self.bert_config.num_hidden_layers - 1

    def linear_interpolation_loss(
        self, hidden_states, original_logits, fast_weights, nr_steps=1
    ):

        original_logits = original_logits.detach()
        mix_layer = self.get_mix_layer()

        hidden_states = hidden_states[-1][mix_layer].split(split_size=1)

        interpolated_hidden = []
        interpolated_logits = []
        nr_samples = min(
            self.args.num_classes_per_set * self.args.num_target_samples, 15
        )
        # indices = np.random.choice(list(range(len(interpolated_hidden))), size=nr_samples)
        candidates = [
            (i, j)
            for i in range(len(hidden_states))
            for j in range(i, len(hidden_states))
        ]
        candidate_ix = list(range(len(candidates)))
        candidate_ix = np.random.choice(candidate_ix, size=nr_samples)
        candidates = [c for i, c in enumerate(candidates) if i in candidate_ix]

        for i, j in candidates:
            for w1 in np.random.beta(a=2, b=2, size=nr_steps):
                h1, y1 = hidden_states[i], original_logits[i]
                h2, y2 = hidden_states[j], original_logits[j]

                interpolated_hidden.append(w1 * h1.squeeze() + (1 - w1) * h2.squeeze())
                interpolated_logits.append(w1 * y1.squeeze() + (1 - w1) * y2.squeeze())

        interpolated_hidden = torch.stack(interpolated_hidden)  # [indices] #TODO: tmp
        interpolated_logits = torch.stack(interpolated_logits)  # [indices]

        logits, _, _, = self.classifier.forward_from(
            hidden_states=interpolated_hidden,
            layer=mix_layer,
            params=fast_weights,
            num_step=self.args.number_of_training_steps_per_iter - 1,
        )

        return self._teacher_KL_loss(
            logits, interpolated_logits, return_nr_correct=False
        )

    def net_forward(
        self,
        x,
        teacher_unary,
        fast_model,
        training,
        num_step,
        return_nr_correct=True,
        mask=None,
        task_name="",
        is_inner_loop=False,
        consistency_training=False,
        aug_x=None,
        aug_mask=None,
    ):

        #####################################################
        # CE loss
        #####################################################

        student_logits, pooled, hidden_states = self.classifier(
            input_ids=x,
            attention_mask=mask,
            num_step=num_step,
            params=fast_model,
            return_pooled=True,  # self.use_triplet_loss,
        )

        set_kl_loss = False
        if task_name in self.gold_label_tasks and self.meta_loss.lower() == "kl":
            set_kl_loss = True
            self.meta_loss = "ce"

        loss, is_correct = self.inner_loss(
            student_logits, teacher_unary, return_nr_correct=return_nr_correct
        )

        if self.scale_losses:
            # Scale loss wrt nr of classes
            num_classes = student_logits.size(-1)
            loss = loss * (self.args.num_classes_per_set / num_classes)

        #####################################################
        # Triplet loss
        #####################################################

        losses = {TOTAL_LOSS_KEY: loss, CE_LOSS_KEY: loss.detach().cpu().item()}
        if self.use_triplet_loss and (
            not is_inner_loop or self.triplet_loss_in_inner_loop
        ):

            losses[CE_LOSS_KEY] = loss.detach().cpu().item()

            _, labels = teacher_unary.max(dim=1)

            triplet_loss = self.triplet_loss_lambda * self.triplet_loss_new(
                pooled,
                labels,
                # hard_pairs
            )
            # logg loss
            losses[TOTAL_LOSS_KEY] += triplet_loss
            losses[TRIPLET_lOSS_KEY] = triplet_loss.detach().cpu().item()

        #####################################################
        # Feature space interpolation loss
        #####################################################
        if self.use_convex_feature_space_loss and (
            not is_inner_loop or self.convex_feature_space_loss_in_inner_loop
        ):
            interpolation_loss = (
                self.convex_feature_space_loss_lambda
                * self.linear_interpolation_loss(
                    hidden_states,
                    student_logits,
                    fast_model,
                    nr_steps=self.convex_feature_space_loss_nr_steps,
                )
            )

            losses[TOTAL_LOSS_KEY] += interpolation_loss
            losses[INTERPOLATION_LOSS_KEY] = interpolation_loss.detach().cpu().item()

        if set_kl_loss:
            self.meta_loss = "kl"
        res = {
            "losses": losses,
            "is_correct": is_correct,
            "logits": student_logits.detach(),
        }
        return res

    def apply_consistency_training(self, fast_model, logits, aug_x, aug_mask, y_true):

        fast_model = {k: v for k, v in fast_model.items() if v is not None}

        aug_student_logits, pooled, _ = self.classifier(
            input_ids=aug_x,
            attention_mask=aug_mask,
            num_step=self.args.number_of_training_steps_per_iter - 1,
            params=fast_model,
            return_pooled=False,
        )

        consistency_loss, ce_loss = self.consistency_loss(
            student_logits=aug_student_logits, teacher_logits=logits, y_true=y_true
        )

        res = {}
        res[CONSISTENCY_LOSS_KEY] = consistency_loss.detach().cpu().item()
        res[CE_LOSS_KEY] = ce_loss.detach().cpu().item()
        res[TOTAL_LOSS_KEY] = consistency_loss + ce_loss

        return res

    def eval_dataset(self, fast_weights, dataloader, to_gpu):
        original_nr_convex_feature_space_loss_steps = int(
            self.convex_feature_space_loss_nr_steps
        )
        # To avoid slow-down due to extra computation of larger batch size during eval
        self.convex_feature_space_loss_nr_steps = 10

        print("Evaluating model...")
        losses = []
        is_correct_preds = []

        if to_gpu:
            self.device = torch.device("cuda")
            self.classifier.to(self.device)

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Evaluating",
                leave=False,
                total=len(dataloader),
            ):
                batch = tuple(t.to(self.device) for t in batch)
                x, mask, y_true = batch

                res = self.net_forward(
                    x,
                    mask=mask,
                    teacher_unary=y_true,
                    fast_model=fast_weights,
                    training=False,
                    return_nr_correct=True,
                    num_step=self.args.number_of_training_steps_per_iter - 1,
                )
                eval_losses = res["losses"]
                is_correct = res["is_correct"]

                loss = {}
                for k in eval_losses.keys():
                    if k == TOTAL_LOSS_KEY:
                        loss[k] = eval_losses[k].item()
                    else:
                        loss[k] = eval_losses[k]
                losses.append(loss)
                is_correct_preds.extend(is_correct.tolist())
        # Set back
        self.convex_feature_space_loss_nr_steps = (
            original_nr_convex_feature_space_loss_steps
        )
        return losses, is_correct_preds

    def inner_update_step(
        self,
        fast_weights,
        num_step,
        x,
        mask,
        y,
        teacher_name,
        use_second_order,
        prototypes=None,
    ):

        torch.cuda.empty_cache()
        if torch.cuda.device_count() > 1:
            torch.cuda.synchronize()

        res = self.net_forward(
            x=x,
            mask=mask,
            num_step=num_step,
            teacher_unary=y,
            fast_model=fast_weights,
            training=True,
            return_nr_correct=True,
            task_name=teacher_name,
            is_inner_loop=True,
        )
        support_losses = res["losses"]
        is_correct = res["is_correct"]

        fast_weights = self.apply_inner_loop_update(
            loss=support_losses[TOTAL_LOSS_KEY],
            names_weights_copy=fast_weights,
            use_second_order=use_second_order,
            current_step_idx=num_step,
            allow_unused=False,
        )

        return fast_weights, support_losses, is_correct

    def split_query_and_aug(
        self, x_target_set_task, len_target_set_task, y_target_set_task, training_phase
    ):

        x_aug_set_task = None
        len_aug_set_task = None
        if self.use_consistency_loss and training_phase:
            num_samples, _ = x_target_set_task.shape
            x_aug_set_task = x_target_set_task[num_samples // 2 :]
            x_target_set_task = x_target_set_task[: num_samples // 2]

            len_aug_set_task = len_target_set_task[num_samples // 2 :]
            len_target_set_task = len_target_set_task[: num_samples // 2]

            y_aug_set_task = y_target_set_task[num_samples // 2 :]
            y_target_set_task = y_target_set_task[: num_samples // 2]

        return (
            x_target_set_task,
            len_target_set_task,
            y_target_set_task,
            x_aug_set_task,
            len_aug_set_task,
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
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
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

        if self.num_freeze_epochs <= epoch:
            self.classifier.unfreeze()

        losses = {TOTAL_LOSS_KEY: 0}
        if self.use_triplet_loss:
            losses[CE_LOSS_KEY] = 0
            losses[TRIPLET_lOSS_KEY] = 0

        if self.use_consistency_loss:
            losses[CE_LOSS_KEY] = 0
            losses[CONSISTENCY_LOSS_KEY] = 0
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

            ##############################################
            # Inner-loop updates
            ##############################################

            for num_step in range(num_steps):

                torch.cuda.empty_cache()
                fast_weights, support_losses, is_correct = self.inner_update_step(
                    x=x_support_set_task,
                    mask=len_support_set_task,
                    num_step=num_step,
                    y=y_support_set_task,
                    fast_weights=fast_weights,
                    teacher_name=teacher_name,
                    use_second_order=use_second_order,
                )

            ##############################################
            # Outer-loop update
            ##############################################
            task_lang_log.append(support_losses[TOTAL_LOSS_KEY].detach().item())
            task_lang_log.append(np.mean(is_correct))
            del support_losses

            context_manager = (
                torch.no_grad()
                if (self.consistency_training and task_id % 2 == 0)
                else contextlib.nullcontext()
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
                else:
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

        self.classifier.eval()
        self.inner_loop_optimizer.eval()
        # Save some computation / memory
        self.inner_loop_optimizer.requires_grad_(False)

        self.inner_loop_optimizer.to(self.device)
        # self.classifier.to(self.device)

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

                batch = tuple(t.to(self.device) for t in batch)
                x, mask, y_true = batch

                ##############################################
                # Inner-loop updates
                ##############################################
                for num_step in range(self.args.number_of_training_steps_per_iter):

                    (
                        names_weights_copy,
                        support_losses,
                        is_correct,
                    ) = self.inner_update_step(
                        x=x,
                        mask=mask,
                        num_step=num_step,
                        y=y_true,
                        fast_weights=names_weights_copy,
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
                        for param_name, param in names_weights_copy.items():
                            writer.add_histogram(
                                task_name + "/" + param_name, param, num_step + 1
                            )
                        writer.flush()

        #########################################################
        # Evaluate finetuned model
        #########################################################
        losses, is_correct_preds = self.eval_dataset(
            fast_weights=names_weights_copy,
            dataloader=dev_dataloader,
            to_gpu=train_on_cpu,
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
