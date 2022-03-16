import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from optimizers.radam import RAdam
from utils.torch_utils import cross_entropy_with_probs
from dataloader import *


class MyDataParallel(nn.DataParallel):
    def __init__(self, model, **kwargs):
        super(MyDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def zero_grad(self, params=None):
        self.module.zero_grad()

    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step):

        return self.module.update_params(
            names_weights_dict=names_weights_dict,
            names_grads_wrt_params_dict=names_grads_wrt_params_dict,
            num_step=num_step,
        )


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class FewShotClassifier(nn.Module):
    def __init__(self, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(FewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.current_epoch = 0
        self.is_distil = "distil" in args.pretrained_weights
        self.is_xlm = "xlm" in args.pretrained_weights
        self.meta_update_method = args.meta_update_method.lower()

        self.meta_loss = args.meta_loss
        self.gold_label_tasks = args.gold_label_tasks

        self.rng = set_torch_seed(seed=args.seed)

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.temperature = args.temperature

        self.task_learning_rate = args.init_inner_loop_learning_rate

        # consistency training
        self.use_consistency_loss = args.use_consistency_loss
        self.consistency_lambda = args.consistency_loss_lambda
        self.consistency_beta = args.consistency_loss_beta

    def get_across_task_loss_metrics(self, total_losses):
        losses = dict()

        losses["loss"] = torch.mean(torch.stack(total_losses))

        return losses

    def inner_loss(self, student_logits, teacher_logits, return_nr_correct=False):

        if self.meta_loss.lower() == "ce":
            _, teacher_preds = teacher_logits.max(dim=1)
            return self._teacher_cross_entropy(
                student_logits, teacher_preds, return_nr_correct
            )
        elif self.meta_loss.lower() == "kl":
            return self._teacher_KL_loss(
                student_logits, teacher_logits, return_nr_correct
            )
        else:
            raise AssertionError("--meta_loss should be either 'ce' or 'kl'.")

    def consistency_loss(self, student_logits, teacher_logits, y_true):

        teacher_logits = teacher_logits.detach()

        # teacher_probs = F.softmax(teacher_logits, dim=-1)
        _, teacher_preds = teacher_logits.max(dim=1)
        _, y_true_index = y_true.max(dim=1)

        correct_classified = teacher_preds == y_true_index

        # KL samples
        if correct_classified.sum() > 0:
            kl_loss = self.consistency_lambda * self._teacher_KL_loss(
                student_logits=student_logits[correct_classified],
                teacher_logits=teacher_logits[correct_classified],
                return_nr_correct=False,
            )
        else:
            kl_loss = 0
        # CE samples
        if (~correct_classified).sum() > 0:
            ce_loss = self._teacher_cross_entropy(
                student_logits=student_logits[~correct_classified],
                teacher_preds=y_true_index[~correct_classified],
                return_nr_correct=False,
            )
        else:
            ce_loss = 0

        return (
            correct_classified.float().mean() * kl_loss,
            (1 - correct_classified.float().mean()) * ce_loss,
        )

    def _teacher_KL_loss(self, student_logits, teacher_logits, return_nr_correct=False):

        loss = (
            self.ce_loss_fct(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
            )
            # * (self.temperature) ** 2
        )

        if torch.isnan(loss):
            print(student_logits, teacher_logits)

        if return_nr_correct:
            _, student_preds = student_logits.max(1)
            _, teacher_preds = teacher_logits.max(1)
            is_correct = (
                teacher_preds.detach().cpu().numpy()
                == student_preds.detach().cpu().numpy()
            )

            return loss, is_correct

        return loss

    def _teacher_cross_entropy(
        self, student_logits, teacher_preds, return_nr_correct=False
    ):

        loss = F.cross_entropy(student_logits, teacher_preds)

        if return_nr_correct:
            _, student_preds = student_logits.max(1)

            is_correct = (teacher_preds == student_preds).detach().cpu().numpy()

            return loss, is_correct

        return loss

    def _pooler_loss(self, student_encodings, teacher_unary, last_layer_only):

        enc_dim = teacher_unary.size(-1)
        if last_layer_only:  # Only compare the final pooler layers
            student_encodings = student_encodings[:, :, -1, :]
            teacher_unary = teacher_unary[:, -1, :]

        student_encodings = student_encodings.view(-1, enc_dim)
        teacher_unary = teacher_unary.view(-1, enc_dim)

        target = student_encodings.new(student_encodings.size(0)).fill_(1)

        loss = self.cosine_loss(student_encodings, teacher_unary, target)

        return loss

    def _unbiased_entropy_loss(self, logits):
        """
        Computes the Entropy of the logits against a uniform distribution over all labels.
        :param logits: pre-softmax activations of model
        :return: loss
        """
        n_samples, n_labels = logits.size()
        init_unbiased_probs = torch.FloatTensor(size=(n_samples, n_labels)).fill_(
            1.0 / n_labels
        )
        init_unbiased_probs = init_unbiased_probs.to(device=self.device)

        init_loss = cross_entropy_with_probs(logits, init_unbiased_probs)
        return init_loss

    def entropy_reduction_loss(self, x, model):

        hidden_states = model(x)[1]
        final_pooler = hidden_states[-1][:, 0]

        student_unary = self.class_head(final_pooler)

        return self._unbiased_entropy_loss(student_unary)

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
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :param last_layer_only: Flag indicating whether loss be computed on the final layer or all layers
        :return: the cosine embedding loss between the teacher- and student encodings
        """
        student_logits = fast_model.forward(
            input_ids=x, attention_mask=mask, num_step=num_step, is_train=training
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

    def finetune_baseline_epoch(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        best_loss,
        eval_every,
        model_save_dir,
        epoch,
    ):
        """
        Finetunes the meta-learned classifier on a dataset
        :param model: base model to "finetune" - DistilBertModel instance
        :param train_dataloader: Dataloader with train examples
        :param dev_dataloader: Dataloader with validation examples
        :param best_loss: best achieved loss on dev set up till now
        :param eval_every: eval on dev set after eval_every updates
        :param model_save_dir: directory to save the model to
        :param task_name: name of the task finetuning is performed on
        :param epoch: current epoch number
        :return: best_loss
        """
        self.device = torch.device("cpu")
        model.to(self.device)
        model.train()

        total_iters = -1
        # TODO: test hyperparams
        optimizer = RAdam(
            model.parameters(), lr=1e-5
        )  # , betas=(0.95, 0.999)) #, momentum=0.9)

        eval_every = (
            eval_every if eval_every < len(train_dataloader) else len(train_dataloader)
        )
        print(len(train_dataloader))
        with tqdm(initial=0, total=eval_every) as pbar_train:

            for batch_idx, batch in enumerate(train_dataloader):

                torch.cuda.empty_cache()
                total_iters += 1
                batch = tuple(t.to(self.device) for t in batch)
                # for j in range(5):
                # Forward pass
                input_ids, teacher_unary = batch
                student_unary = model(input_ids)[0]

                # backward pass
                loss = self.inner_loss(student_unary, teacher_unary)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                pbar_train.update(1)
                pbar_train.set_description(
                    "finetuning phase {} -> loss: {}".format(
                        total_iters + 1, loss.item()
                    )
                )

                # Evaluation
                if (total_iters + 1) % eval_every == 0:
                    print("Evaluating model...")
                    self.device = torch.device("cuda")
                    model.to(self.device)
                    model.eval()
                    losses = []
                    is_correct_preds = []
                    with torch.no_grad():
                        for batch in tqdm(
                            dev_dataloader,
                            desc="Evaluating",
                            leave=False,
                            total=len(dev_dataloader),
                        ):
                            batch = tuple(t.to(self.device) for t in batch)

                            input_ids, teacher_unary = batch
                            student_unary = model(input_ids)[0]

                            loss, is_correct = self.inner_loss(
                                student_unary, teacher_unary, True
                            )
                            is_correct_preds.extend(is_correct.tolist())
                            losses.append(loss.item())

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

                        model.save_pretrained(os.path.join(model_save_dir, str(epoch)))

                    return model, best_loss, avg_loss, accuracy

    #################################################################################
    # METHODS TO OVERWRITE
    #################################################################################

    def forward(
        self,
        data_batch,
        epoch,
        use_second_order,
        use_multi_step_loss_optimization,
        num_steps,
        training_phase,
    ):
        """
         Performs a full episode of meta-learning
        :param data_batch: A data batch containing the support and target sets.
         :param epoch: Current epoch's index
         :param use_second_order: A boolean saying whether to use second order derivatives.
         :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
         target loss (True) or whether to use multi step loss which improves the stability of the system (False)
         :param num_steps: Number of inner loop steps.
         :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
         :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        raise NotImplementedError

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
        **kwargs
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
        raise NotImplementedError

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            use_second_order=self.args.second_order
            and epoch > self.args.first_order_to_second_order_epoch,
            num_steps=self.args.number_of_training_steps_per_iter,
            training_phase=True,
        )
        return losses

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            use_second_order=False,
            num_steps=self.args.number_of_evaluation_steps_per_iter,
            training_phase=False,
        )

        return losses

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        for k in [
            SUPPORT_SET_SAMPLES_KEY,
            SUPPORT_SET_LENS_KEY,
            SUPPORT_SET_ENCODINGS_KEY,
            TARGET_SET_SAMPLES_KEY,
            TARGET_SET_LENS_KEY,
            TARGET_SET_ENCODINGS_KEY,
            CLASS_NAMES_KEY,
            CLASS_NAMES_LENS_KEY,
            CLASS_NAMES_ENCODING_KEY,
        ]:
            if k in data_batch.keys():
                data_batch[k] = [x.to(self.device) for x in data_batch[k]]

        # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
        losses, task_lang_log = self.train_forward_prop(
            data_batch=data_batch, epoch=epoch
        )

        # print(prof.key_averages().table(sort_by='self_cuda_memory_usage'))

        self.optimizer.step()

        losses["learning_rate"] = self.optimizer.current_lr
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, task_lang_log

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        set_meta_loss_back = False
        if self.meta_loss.lower() == "kl" and self.args.val_using_cross_entropy:
            # Use cross entropy on gold labels as no teacher encoding is available
            self.meta_loss = "ce"
            set_meta_loss_back = True

        if self.training:
            self.eval()

        for k in [
            SUPPORT_SET_SAMPLES_KEY,
            SUPPORT_SET_LENS_KEY,
            SUPPORT_SET_ENCODINGS_KEY,
            TARGET_SET_SAMPLES_KEY,
            TARGET_SET_LENS_KEY,
            TARGET_SET_ENCODINGS_KEY,
            CLASS_NAMES_KEY,
            CLASS_NAMES_LENS_KEY,
            CLASS_NAMES_ENCODING_KEY,
        ]:
            if k in data_batch.keys():
                data_batch[k] = [x.to(self.device) for x in data_batch[k]]

        losses = self.evaluation_forward_prop(
            data_batch=data_batch, epoch=self.current_epoch
        )
        if set_meta_loss_back:
            self.meta_loss = "kl"
        return losses

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state["network"] = self.state_dict()
        state["optimizer"] = self.optimizer.state_dict()

        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath, map_location=torch.device("cpu"))
        state_dict_loaded = state["network"]

        self.load_state_dict(state_dict=state_dict_loaded, strict=True)
        # self.optimizer.load_state_dict(state["optimizer"])
        return state
