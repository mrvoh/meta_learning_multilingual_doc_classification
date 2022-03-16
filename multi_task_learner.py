import torch
import numpy as np
from torch import nn, optim
from transformers import AutoModel, AutoConfig
from copy import deepcopy

from few_shot_learning_system import *
from maml import TOTAL_LOSS_KEY, CE_LOSS_KEY
from ranger import Ranger
from tqdm import tqdm


class MultiTaskLearner(FewShotClassifier):
    def __init__(self, device, args):

        super(MultiTaskLearner, self).__init__(device, args)

        # Init model
        config = AutoConfig.from_pretrained(args.pretrained_weights)
        config.num_labels = args.num_classes_per_set
        self.bert_config = config
        self.encoder = AutoModel.from_pretrained(args.pretrained_weights, config=config)

        try:
            dim = config.dim
        except AttributeError:
            dim = config.hidden_size

        self.class_heads = nn.ModuleDict(
            {
                "MLDocTrain": nn.Linear(dim, 4),
                "AmazonTrain": nn.Linear(dim, 2),
                "TREC": nn.Linear(dim, 6),
                "Yelp": nn.Linear(dim, 2),
                "IMDB": nn.Linear(dim, 2),
                "IMPHateSpeech": nn.Linear(dim, 2),
                "FinancialPhraseBank": nn.Linear(dim, 3),
                "SciCite": nn.Linear(dim, 3),
                "FewRel": nn.Linear(dim, 64),
                "LEDGAR": nn.Linear(dim, 181),
                "DBPedia": nn.Linear(dim, 219),
                "Huffpost": nn.Linear(dim, 41),
            }
        )

        self.hidden_dim = dim

        self.encoder.to(self.device)
        self.class_heads.to(self.device)
        # Init optimizer
        self.optimizer = Ranger(
            self.parameters(),
            lr=args.meta_learning_rate,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs * self.args.total_iter_per_epoch,
            eta_min=self.args.min_learning_rate,
        )

    def forward(
        self,
        data_batch,
        epoch,
        use_second_order,
        num_steps,
        training_phase,
    ):

        x_support_set = data_batch[SUPPORT_SET_SAMPLES_KEY]
        len_support_set = data_batch[SUPPORT_SET_LENS_KEY]
        x_target_set = data_batch[TARGET_SET_SAMPLES_KEY]
        len_target_set = data_batch[TARGET_SET_LENS_KEY]
        y_support_set = data_batch[SUPPORT_SET_ENCODINGS_KEY]
        y_target_set = data_batch[TARGET_SET_ENCODINGS_KEY]
        teacher_names = data_batch[SELECTED_CLASS_KEY]

        meta_batch_size = self.args.batch_size
        self.zero_grad()
        self.train()
        task_lang_logs = []
        task_accuracies = []
        losses = {TOTAL_LOSS_KEY: 0, CE_LOSS_KEY: 0}

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

            output = self.encoder(x_support_set_task, len_support_set_task)
            if len(output) == 1:
                # Distilbert has no pooler
                pooled = output[0][:, 0, :]
            else:
                # Bert-like models
                pooled = output[1]

            task_name = teacher_name.split("_")[0]  # remove language extension
            class_head = self.class_heads[task_name]

            logits = class_head(pooled)

            loss, is_correct = self.inner_loss(
                student_logits=logits,
                teacher_logits=y_support_set_task,
                return_nr_correct=True,
            )

            loss = loss / meta_batch_size
            loss.backward()

            accuracy = np.mean(is_correct)
            task_accuracies.append(accuracy)
            # store statistics
            task_lang_log.append(loss.detach().item())
            task_lang_log.append(accuracy)

            losses[TOTAL_LOSS_KEY] += loss.detach().item()

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

        encoder = deepcopy(self.encoder)

        # Only hold a copy of the model on GPU
        self.encoder.to(torch.device("cpu"))
        self.class_heads.to(torch.device("cpu"))

        if train_on_cpu:
            self.device = torch.device("cpu")

        encoder.to(self.device)

        encoder.eval()

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

                # Initialize classification head
                if batch_idx == 0:
                    task_name = task_name.split("_")[0]  # remove language extension
                    if task_name in self.class_heads.keys():
                        class_head = deepcopy(self.class_heads[task_name])
                    elif "Amazon" in task_name:
                        class_head = deepcopy(self.class_heads["AmazonTrain"])
                    elif "MLDoc" in task_name:
                        class_head = deepcopy(self.class_heads["MLDocTrain"])
                    else:
                        class_head = nn.Linear(self.hidden_dim, y_true.size(1))

                    class_head.to(self.device)

                    optimizer = Ranger(
                        [
                            {
                                "params": encoder.parameters(),
                                "lr": self.args.init_inner_loop_learning_rate,
                            },
                            {
                                "params": class_head.parameters(),
                                "lr": self.args.init_inner_loop_learning_rate,
                            },
                        ],
                        lr=self.args.init_inner_loop_learning_rate,
                    )
                for num_step in range(self.args.number_of_training_steps_per_iter):
                    # Update the model
                    output = encoder(x, mask)
                    if len(output) == 1:
                        # Distilbert has no pooler
                        pooled = output[0][:, 0, :]
                    else:
                        # Bert-like models
                        pooled = output[1]

                    logits = class_head(pooled)

                    loss, is_correct = self.inner_loss(
                        student_logits=logits,
                        teacher_logits=y_true,
                        return_nr_correct=True,
                    )

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    encoder.zero_grad()
                    class_head.zero_grad()

                    pbar_train.update(1)
                    desc = f"finetuning phase {batch_idx * self.args.number_of_training_steps_per_iter + num_step + 1} -> loss: {loss.item()}"

                    pbar_train.set_description(desc)

        #########################################################
        # Evaluate
        #########################################################
        print("Evaluating model...")
        losses = []
        is_correct_preds = []

        if train_on_cpu:
            self.device = torch.device("cuda")
            encoder.to(self.device)
            class_head.to(self.device)

        with torch.no_grad():
            for batch in tqdm(
                dev_dataloader,
                desc="Evaluating",
                leave=False,
                total=len(dev_dataloader),
            ):
                batch = tuple(t.to(self.device) for t in batch)
                x, mask, y_true = batch

                output = encoder(x, mask)
                if len(output) == 1:
                    # Distilbert has no pooler
                    pooled = output[0][:, 0, :]
                else:
                    # Bert-like models
                    pooled = output[1]

                logits = class_head(pooled)

                loss, is_correct = self.inner_loss(
                    student_logits=logits, teacher_logits=y_true, return_nr_correct=True
                )

                losses.append(loss.item())
                is_correct_preds.extend(is_correct.tolist())

        # Compute avg
        avg_loss = {TOTAL_LOSS_KEY: np.mean(losses)}
        accuracy = np.mean(is_correct_preds)

        print("Accuracy", accuracy)
        if avg_loss[TOTAL_LOSS_KEY] < best_loss:
            best_loss = avg_loss[TOTAL_LOSS_KEY]
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

        self.encoder.to(self.device)
        self.class_heads.to(self.device)

        return None, best_loss, avg_loss, accuracy
