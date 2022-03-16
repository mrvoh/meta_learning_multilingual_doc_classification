from dataloader import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from maml import MAMLFewShotClassifier
from protomaml import ProtoMAMLFewShotClassifier
from protomaml_threeway import ThreeWayProtoMAMLFewShotClassifier
from reptile import ReptileFewShotClassifier
from prototypical_network import PrototypicalNetworkFewShotClassifier
from multi_task_learner import MultiTaskLearner
from utils.parser_utils import get_args
import os

import torch


if __name__ == "__main__":
    # Combines the arguments, model, data and experiment builders to run an experiment
    os.environ[
        "DATASET_DIR"
    ] = "/home/mrvoh/Desktop/datasets/multi_task_class_descr"  # "/home/mrvoh/Desktop/datasets/FewRel/fewrel_standard_split" "/media/mrvoh/ubuntu-usb2/data/lang_gen/XNLI/gold_low_resource/ar_bg_fr_ru_sw_th"#  "/media/mrvoh/ubuntu-usb2/data/lang_gen/TextClass/bronze/text_class_grid_search (1)" #/home/mrvoh/Desktop/RCV2_in_domain" #"/home/mrvoh/Documents/source/thesis/data_utils/opus_mldoc" #r   #"/home/mrvoh/Desktop/text_class_lang_gen/" for testing
    args, device = get_args()

    # device = torch.device('cpu')
    update_method = args.meta_update_method.lower()

    if update_method == "maml":
        model = MAMLFewShotClassifier(args=args, device=device)
    elif update_method == "protomaml":
        model = ProtoMAMLFewShotClassifier(args=args, device=device)
    elif update_method == "threewayprotomaml":
        model = ThreeWayProtoMAMLFewShotClassifier(args=args, device=device)
        print(
            "Doubling the amount of samples in the support set to split prototype creation and fast adapation..."
        )
        args.num_samples_per_class *= 2
    elif update_method == "reptile":
        model = ReptileFewShotClassifier(args=args, device=device)
    elif "prototypical" in update_method.lower():
        model = PrototypicalNetworkFewShotClassifier(args=args, device=device)
    elif update_method == "mtl":
        model = MultiTaskLearner(args=args, device=device)
    else:
        raise AssertionError(
            "The meta update method must be chosen from [maml, protomaml, threewayprotomaml, reptile, prototypical]"
        )

    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
