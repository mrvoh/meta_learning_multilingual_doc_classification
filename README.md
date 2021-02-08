# Multilingual and cross-lingual document classification: A meta-learning approach
This repository is based on the work of Antreas Antoniou, [How To Train Your MAML](https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch).
## Contents
| Section | Description |
|-|-|
| [Setup](#setup) | How to setup a working environment |
| [Architectures](#architectures) | Available models |
| [Data and Preprocessing](#data-and-preprocessing) | How to prepare and utilize a (custom) dataset |
| [Training/evaluating](#training/evaluating) | Train and evaluate a model |
| [Using FastText](#using-fasttext) | Create custom word vectors or baseline model |
| [Using ULMFiT](#using-ulmfit) | Using ULMFiT as word encoder and to create document encodings |
| [Hyperparameter optimization](#hyperparameter-optimization) | Finding optimal hyperparameters using [Hyperopt](https://github.com/hyperopt/hyperopt) |
| [References](#references) | References for this repo | 

## Setup

[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Create virtual environment:
```
conda create --name meta python=3.8
conda activate hcapsnet
```
[3]
Install PyTorch (>1.5). Please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the specifics for your platform.

[4] Clone the repository:
```
git clone https://github.com/mrvoh/meta_learning_multilingual_doc_classification.git
cd meta_learning_multilingual_doc_classification
```
[5] Install the Ranger optimizer
Instructions found in [the original repo on Github](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
[6] Install the requirements:
```
pip install -r requirements.txt
```

## Data pre-processing

## Running an experiment



## Citation

Please cite our [paper](https://arxiv.org/abs/2101.11302) if you use it in your own work.
```bibtex
@article{van2021multilingual,
  title={Multilingual and cross-lingual document classification: A meta-learning approach},
  author={van der Heijden, Niels and Yannakoudakis, Helen and Mishra, Pushkar and Shutova, Ekaterina},
  journal={arXiv preprint arXiv:2101.11302},
  year={2021}
}
```