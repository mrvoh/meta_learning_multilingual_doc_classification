def get_args():
    import argparse
    import os
    import torch
    import json

    parser = argparse.ArgumentParser(
        description="Welcome to the MAML++ training and inference system"
    )

    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="Batch_size for experiment",
    )
    parser.add_argument("--reset_stored_filepaths", type=str, default="False")
    parser.add_argument("--num_of_gpus", type=int, default=1)
    parser.add_argument(
        "--indexes_of_folders_indicating_class", nargs="+", default=[-2, -3]
    )
    parser.add_argument(
        "--train_val_test_split",
        nargs="+",
        default=[0.73982737361, 0.26, 0.13008631319],
    )
    parser.add_argument("--samples_per_iter", nargs="?", type=int, default=1)
    parser.add_argument("--seed", type=int, default=104)

    parser.add_argument("--gpu_to_use", type=int)
    parser.add_argument("--num_dataprovider_workers", nargs="?", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default="omniglot_dataset")
    parser.add_argument("--dataset_path", type=str, default="datasets/omniglot_dataset")
    parser.add_argument(
        "--pretrained_weights", type=str, default="distilbert-base-multilingual-cased"
    )
    parser.add_argument("--reset_stored_paths", type=str, default="False")
    parser.add_argument(
        "--experiment_name",
        nargs="?",
        type=str,
    )
    parser.add_argument(
        "--continue_from_epoch",
        nargs="?",
        type=str,
        default="latest",
        help="Continue from checkpoint of epoch",
    )
    parser.add_argument(
        "--num_target_samples", type=int, default=15, help="Dropout_rate_value"
    )
    parser.add_argument(
        "--second_order", type=str, default="False", help="Dropout_rate_value"
    )
    parser.add_argument(
        "--total_epochs", type=int, default=200, help="Number of epochs per experiment"
    )
    parser.add_argument(
        "--total_iter_per_epoch",
        type=int,
        default=500,
        help="Number of iters per epoch",
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=0.00001, help="Min learning rate"
    )
    parser.add_argument(
        "--meta_learning_rate",
        type=float,
        default=0.00002,
        help="Learning rate of base learner",
    )
    parser.add_argument(
        "--meta_inner_optimizer_learning_rate",
        type=float,
        default=0.001,
        help="Learning rate of base learner",
    )

    parser.add_argument(
        "--num_classes_per_set",
        type=int,
        default=2,
        help="Number of classes to sample per set",
    )
    parser.add_argument(
        "--number_of_training_steps_per_iter",
        type=int,
        default=1,
        help="Number of classes to sample per set",
    )
    parser.add_argument(
        "--number_of_evaluation_steps_per_iter",
        type=int,
        default=1,
        help="Number of classes to sample per set",
    )
    parser.add_argument(
        "--num_samples_per_class",
        type=int,
        default=1,
        help="Number of samples per set to sample",
    )
    parser.add_argument(
        "--eval_using_full_task_set",
        action="store_true",
        help="Whether to evaluate validation performance on the full task set or a limited query set",
    )
    parser.add_argument(
        "--split_support_and_query",
        action="store_false",
    )
    parser.add_argument(
        "--sample_task_to_size_ratio",
        action="store_true",
        help="Whether to sample episodes from tasks per ratio of the total task size",
    )
    parser.add_argument(
        "--enable_inner_loop_optimizable_bn_params",
        action="store_false",
        help="Whether to sample episodes from tasks per ratio of the total task size",
    )

    parser.add_argument(
        "--shuffle_labels",
        action="store_false",
        help="Whether to shuffle labels within an episode as means of data augmentation.",
    )


    parser.add_argument(
        "--num_evaluation_seeds",
        type=int,
        default=1,
        help="Number of evaluation seeds to run when evaluating on the full validation set",
    )

    parser.add_argument(
        "--init_class_head_lr_multiplier",
        type=float,
        default=1,
        help="Factor to multiply init lr for class head",
    )
    parser.add_argument(
        "--gold_label_task_sample_ratio",
        type=float,
        default=0,
        help="Percentage of tasks sampled only from gold_label_tasks",
    )

    parser.add_argument(
        "--num_freeze_epochs",
        type=int,
        default=0,
        help="Number of epochs to keep the encoder frozen: in meta-updates only the classification head is considered",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs no improvements are allowed before applying early stopping",
    )
    parser.add_argument(
        "--num_start_epochs",
        type=int,
        default=0,
        help="Number of epochs to train without validation",
    )

    parser.add_argument(
        "--protomaml_do_centralize",
        action="store_true",
        help="Whether to apply centralization to prototypes",
    )
    parser.add_argument(
        "--name_of_args_json_file",
        type=str,
        default="experiment_config/rcv2_distill-rcv2_8_4_0.0001_2_42.json",
    )
    parser.add_argument("--val_using_cross_entropy", action="store_true")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Softmax temperature for smoothing distribution of teacher logits.",
    )
    parser.add_argument(
        "--meta_loss",
        type=str,
        default="CE",
        help="Kind of loss to use on teacher encodings: cross entropy of or KL divergence",
    )
    parser.add_argument("--gold_label_tasks", nargs="+", default=[])

    ######################################################################################
    # New parameters for follow-up research
    ######################################################################################
    parser.add_argument(
        "--scale_losses",
        action="store_true",
        help="Whether to scale the losses according to the number of classes in the task",
    )
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="Whether to use stochastic weight averaging during evaluation -- only for ThreewayProtoMAML",
    )
    parser.add_argument(
        "--num_swa", type=float, default=5, help="Margin for the triplet loss"
    )
    parser.add_argument(
        "--use_majority_vote",
        action="store_true",
        help="Whether to use majority voting during evaluation -- only for ThreewayProtoMAML",
    )
    parser.add_argument(
        "--num_majority_votes",
        type=float,
        default=5,
        help="Margin for the triplet loss",
    )
    parser.add_argument(
        "--majority_vote_at_test_only",
        action="store_true",
        help="Whether to use majority voting during testing only -- only for ThreewayProtoMAML",
    )

    parser.add_argument(
        "--use_label_guided_learning",
        action="store_true",
        help="Create prototypes based on label names",
    )

    parser.add_argument(
        "--use_uncertainty_task_weighting",
        action="store_true",
        help="Learn a weight per task based on homoscedastic uncertainty - https://arxiv.org/abs/1705.07115",
    )

    parser.add_argument(
        "--use_triplet_loss",
        action="store_true",
    )
    parser.add_argument(
        "--triplet_loss_in_inner_loop",
        action="store_true",
    )
    parser.add_argument(
        "--use_cosine_distance",
        action="store_true",
    )
    parser.add_argument(
        "--triplet_loss_margin",
        type=float,
        default=0.5,
        help="Margin for the triplet loss",
    )
    parser.add_argument(
        "--triplet_loss_lambda",
        type=float,
        default=1.0,
        help="Scaling factor for the triplet loss",
    )
    parser.add_argument(
        "--use_consistency_loss",
        action="store_true",
    )
    parser.add_argument(
        "--use_multilingual_consistency_loss",
        action="store_true",
    )
    parser.add_argument(
        "--consistency_loss_lambda",
        type=float,
        default=1.0,
        help="Weighting factor for consistency loss",
    )
    parser.add_argument(
        "--consistency_loss_beta",
        type=float,
        default=1.0,
        help="Min certainty for a sample to be considered in consistency training",
    )

    parser.add_argument(
        "--use_convex_feature_space_loss",
        action="store_true",
    )
    parser.add_argument(
        "--convex_feature_space_loss_in_inner_loop",
        action="store_true",
    )
    parser.add_argument(
        "--convex_feature_space_loss_lambda",
        type=float,
        default=1.0,
        help="Weighting factor for consistency loss",
    )
    parser.add_argument(
        "--convex_feature_space_loss_nr_steps",
        type=int,
        default=10,
        help="Weighting factor for consistency loss",
    )

    parser.add_argument(
        "--use_adapter",
        action="store_true",
        help="Use Adapters to train instead of full Transformer model",
    )

    parser.add_argument(
        "--proportion_intra_task_sampling",
        type=float,
        default=0.85,
        help="Discount factor for prototype similarity loss",
    )

    parser.add_argument(
        "--variable_nr_classes_and_samples",
        action="store_true",
    )

    ############################################################################
    # Finetune args
    ############################################################################
    parser.add_argument("--finetune_task_name", type=str, default="val/RCV2_ja_gt")
    parser.add_argument("--num_finetune_epochs", type=int, default=1)
    parser.add_argument("--eval_every_finetune", type=int, default=1)
    parser.add_argument("--finetune_base_model", action="store_true")
    parser.add_argument("--percentage_train_finetune", type=float, default=0.95)
    parser.add_argument("--factory_finetune", action="store_true")
    parser.add_argument("--finetune_on_cpu", action="store_true")
    parser.add_argument(
        "--factory_finetune_tasks_path",
        type=str,
        default="finetune_tasks.txt",
    )
    parser.add_argument(
        "--finetune_out_path",
        type=str,
        default="finetune_log_old.json",
    )
    parser.add_argument(
        "--bootstrap_finetune",
        action="store_true",
        help="Finetunes the fast learner on multiple batches and saves each instance without evaluation",
    )
    parser.add_argument(
        "--num_bootstrap_seeds",
        type=int,
        default=5,
        help="Number of batches to finetune the fast learner on when using bootstrap_finetune",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    if args.name_of_args_json_file is not "None":
        args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            args_dict[key] = os.path.join(os.environ["DATASET_DIR"], args_dict[key])
            print(key, os.path.join(os.environ["DATASET_DIR"], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)

    args.use_cuda = torch.cuda.is_available()

    if args.gpu_to_use == -1:
        args.use_cuda = False

    if args.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.use_swa and args.use_majority_vote:
        print(
            "Both --use_swa and --use_majority_vote are set, defaulting to majority vote."
        )

    return args, device


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, args_dict):
    import json

    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" in key:
            pass
        elif "gpu_to_use" in key:
            pass
        else:
            args_dict[key] = summary_dict[key]

    return args_dict
