from dataloader import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from maml import MAMLFewShotClassifier
from protomaml import ProtoMAMLFewShotClassifier
from reptile import ReptileFewShotClassifier
from prototypical_network import PrototypicalNetworkFewShotClassifier
from utils.parser_utils import get_args
import os


if __name__ == "__main__":
    # Combines the arguments, model, data and experiment builders to run an experiment
    os.environ["DATASET_DIR"] = "/home/mrvoh/Desktop/datasets"
    args, device = get_args()

    update_method = args.meta_update_method.lower()

    if update_method == "maml":
        model = MAMLFewShotClassifier(args=args, device=device)
    elif update_method == "protomaml":
        model = ProtoMAMLFewShotClassifier(args=args, device=device)
    elif update_method == "reptile":
        model = ReptileFewShotClassifier(args=args, device=device)
    elif "prototypical" in update_method.lower():
        model = PrototypicalNetworkFewShotClassifier(args=args, device=device)
    else:
        raise AssertionError(
            "The meta update method must be chosen from [maml, protomaml, threewayprotomaml, reptile, prototypical]"
        )

    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
