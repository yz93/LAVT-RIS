# LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
Welcome to the official repository for the method presented in
"LAVT: Language-Aware Vision Transformer for Referring Image Segmentation."


![Pipeline Image](pipeline.jpg)

Code in this repository is written using [PyTorch](https://pytorch.org/) and is organized in the following way (assuming the working directory is the root directory of this repository):
* `./lib` contains files implementing the main network.
* Inside `./lib`, `_utils.py` defines the highest-level model, which incorporates the backbone network
defined in `backbone.py` and the simple mask decoder defined in `mask_predictor.py`.
`segmentation.py` provides the model interface and initialization functions.
* `./bert` contains files migrated from [Hugging Face Transformers v3.0.2](https://huggingface.co/transformers/v3.0.2/quicktour.html),
which implement the BERT language model.
We used Transformers v3.0.2 during development but it had a bug that would appear when using `DistributedDataParallel`.
Therefore we maintain a copy of the relevant source files in this repository.
This way, the bug is fixed and code in this repository is self-contained.
* `./train.py` is invoked to train the model.
* `./test.py` is invoked to run inference on the evaluation subsets after training.
* `./refer` contains data pre-processing code and is also where data should be placed, including the images and all annotations.
It is cloned from [refer](https://github.com/lichengunc/refer). 
* `./data/dataset_refer_bert.py` is where the dataset class is defined.
* `./utils.py` defines functions that track training statistics and setup
functions for `DistributedDataParallel`.


## Updates
**April 13<sup>th</sup>, 2023**. Using the Dice loss instead of the cross-entropy loss can improve results. Will add code and release weights later when get a chance.

**June 21<sup>st</sup>, 2022**. Uploaded the training logs and trained
model weights of lavt_one.

**June 9<sup>th</sup>, 2022**.
Added a more efficient implementation of LAVT.
* To train this new model, specify `--model` as `lavt_one`
(and `lavt` is still valid for specifying the old model).
The rest of the configuration stays unchanged.
* The difference between this version and the previous one
is that the language model has been moved inside the overall model,
so that `DistributedDataParallel` needs to be applied only once.
Applying it twice (on the standalone language model and the main branch)
as done in the old implementation led to low GPU utility,
which slowed down training.
We recommend training this model on 8 GPUs
(and same as before with batch size 32).

## Setting Up
### Preliminaries
The code has been verified to work with PyTorch v1.7.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.
### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n lavt python==3.7
conda activate lavt
```

2. Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```

### Datasets
1. Follow instructions in the `./refer` directory to set up subdirectories
and download annotations.
This directory is a git clone (minus two data files that we do not need)
from the [refer](https://github.com/lichengunc/refer) public API.

2. Download images from [COCO](https://cocodataset.org/#download).
Please use the first downloading link *2014 Train images [83K/13GB]*, and extract
the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
and put the `pth` file in `./pretrained_weights`.
These weights are needed for training to initialize the model.

### Trained Weights of LAVT for Testing
1. Create the `./checkpoints` directory where we will be storing the weights.
```shell
mkdir ./checkpoints
```
2. Download LAVT model weights (which are stored on Google Drive) using links below and put them in `./checkpoints`.

| [RefCOCO](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing) | [RefCOCO+](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing) | [G-Ref (UMD)](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing) | [G-Ref (Google)](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing) |
|---|---|---|---|

3. Model weights and training logs of the new lavt_one implementation are below.

| RefCOCO | RefCOCO+ | G-Ref (UMD) | G-Ref (Google) |
|:-----:|:-----:|:-----:|:-----:|
|[log](https://drive.google.com/file/d/1YIojIHqe3bxxsWOltifa2U9jH67hPHLM/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1xFMEXr6AGU97Ypj1yr8oo00uObbeIQvJ/view?usp=sharing)|[log](https://drive.google.com/file/d/1Z34T4gEnWlvcSUQya7txOuM0zdLK7MRT/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1HS8ZnGaiPJr-OmoUn4-4LVnVtD_zHY6w/view?usp=sharing)|[log](https://drive.google.com/file/d/14VAgahngOV8NA6noLZCqDoqaUrlW14v8/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/14g8NzgZn6HzC6tP_bsQuWmh5LnOcovsE/view?usp=sharing)|[log](https://drive.google.com/file/d/1JBXfmlwemWSvs92Rky0TlHcVuuLpt4Da/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1IJeahFVLgKxu_BVmWacZs3oUzgTCeWcz/view?usp=sharing)|

* The Prec@K, overall IoU and mean IoU numbers in the training logs will differ
from the final results obtained by running `test.py`,
because only one out of multiple annotated expressions is
randomly selected and evaluated for each object during training.
But these numbers give a good idea about the test performance.
The two should be fairly close.


## Training
We use `DistributedDataParallel` from PyTorch.
The released `lavt` weights were trained using 4 x 32G V100 cards (max mem on each card was about 26G).
The released `lavt_one` weights were trained using 8 x 32G V100 cards (max mem on each card was about 13G).
Using more cards was to accelerate training.
To run on 4 GPUs (with IDs 0, 1, 2, and 3) on a single node:
```shell
mkdir ./models

mkdir ./models/refcoco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output

mkdir ./models/refcoco+
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco+ --model_id refcoco+ --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco+/output

mkdir ./models/gref_umd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcocog --splitBy umd --model_id gref_umd --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/gref_umd/output

mkdir ./models/gref_google
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcocog --splitBy google --model_id gref_google --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/gref_google/output
```
* *--model* is a pre-defined model name. Options include `lavt` and `lavt_one`. See [Updates](#updates).
* *--dataset* is the dataset name. One can choose from `refcoco`, `refcoco+`, and `refcocog`.
* *--splitBy* needs to be specified if and only if the dataset is G-Ref (which is also called RefCOCOg).
`umd` identifies the UMD partition and `google` identifies the Google partition.
* *--model_id* is the model name one should define oneself (*e.g.*, customize it to contain training/model configurations, dataset information, experiment IDs, *etc*.).
It is used in two ways: Training log will be saved as `./models/[args.model_id]/output` and the best checkpoint will be saved as `./checkpoints/model_best_[args.model_id].pth`.
* *--swin_type* specifies the version of the Swin Transformer.
One can choose from `tiny`, `small`, `base`, and `large`. The default is `base`.
* *--pretrained_swin_weights* specifies the path to pre-trained Swin Transformer weights used for model initialization.
* Note that currently we need to manually create the `./models/[args.model_id]` directory via `mkdir` before running `train.py`.
This is because we use `tee` to redirect `stdout` and `stderr` to `./models/[args.model_id]/output` for logging.
This is a nuisance and should be resolved in the future, *i.e.*, using a proper logger or a bash script for initiating training.

## Testing
For RefCOCO/RefCOCO+, run one of
```shell
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
python test.py --model lavt --swin_type base --dataset refcoco+ --split val --resume ./checkpoints/refcoco+.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```
* *--split* is the subset to evaluate, and one can choose from `val`, `testA`, and `testB`.
* *--resume* is the path to the weights of a trained model.

For G-Ref (UMD)/G-Ref (Google), run one of
```shell
python test.py --model lavt --swin_type base --dataset refcocog --splitBy umd --split val --resume ./checkpoints/gref_umd.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
python test.py --model lavt --swin_type base --dataset refcocog --splitBy google --split val --resume ./checkpoints/gref_google.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```
* *--splitBy* specifies the partition to evaluate.
One can choose from `umd` or `google`.
* *--split* is the subset (according to the specified partition) to evaluate, and one can choose from `val` and `test` for the UMD partition, and only `val` for the Google partition..
* *--resume* is the path to the weights of a trained model.

## Results
1. The evaluation results (those reported in the paper) of LAVT trained with a cross-entropy loss and based on our original implementation are summarized as follows:

|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:--------:|
| RefCOCO val     | 84.46 | 80.90 | 75.28 | 64.71 | 34.30 |    72.73    |   74.46  |
| RefCOCO test A  | 88.07 | 85.17 | 79.90 | 68.52 | 35.69 |    75.82    |   76.89  |
| RefCOCO test B  | 79.12 | 74.94 | 69.17 | 59.37 | 34.45 |    68.79    |   70.94  |
| RefCOCO+ val    | 74.44 | 70.91 | 65.58 | 56.34 | 30.23 |    62.14    |   65.81  |
| RefCOCO+ test A | 80.68 | 77.96 | 72.90 | 62.21 | 32.36 |    68.38    |   70.97  |
| RefCOCO+ test B | 65.66 | 61.85 | 55.94 | 47.56 | 27.24 |    55.10    |   59.23  |
| G-Ref val (UMD) | 70.81 | 65.28 | 58.60 | 47.49 | 22.73 |    61.24    |   63.34  |
| G-Ref test (UMD)| 71.54 | 66.38 | 59.00 | 48.21 | 23.10 |    62.09    |   63.62  |
|G-Ref val (Goog.)| 71.16 | 67.21 | 61.76 | 51.98 | 27.30 |    60.50    |   63.66  |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- We have validated LAVT on RefCOCO with multiple runs. The overall IoU on the val set generally lies in the range of 72.73±0.5%.

2. In the following, we report the results of LAVT trained with a multi-class Dice loss and based on the new implementation (`lavt_one`).

|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:--------:|
| RefCOCO val     | 85.87 | 82.13 | 76.64 | 65.45 | 35.30 |    73.50    |   75.41  |
| RefCOCO test A  | 88.47 | 85.63 | 80.57 | 68.84 | 35.71 |    75.97    |   77.31  |
| RefCOCO test B  | 80.20 | 76.49 | 70.34 | 60.12 | 34.94 |    69.33    |   71.86  |
| RefCOCO+ val    | 76.19 | 72.27 | 66.82 | 56.87 | 30.15 |    63.79    |   67.65  |
| RefCOCO+ test A | 82.50 | 79.44 | 74.00 | 63.27 | 31.99 |    69.79    |   72.53  |
| RefCOCO+ test B | 68.03 | 63.35 | 57.29 | 47.92 | 26.98 |    56.49    |   61.22  |
| G-Ref val (UMD) | 75.82 | 71.06 | 63.99 | 52.98 | 27.31 |    64.02    |   67.41  |
| G-Ref test (UMD)| 76.12 | 71.13 | 64.58 | 53.62 | 28.03 |    64.49    |   67.45  |
|G-Ref val (Goog.)| 72.57 | 68.65 | 63.09 | 53.33 | 28.14 |    61.31    |   64.84  |


## Demo: Try LAVT on Your Own Image-Text Pairs
You can run inference on any image-text pair
and visualize the result by running the script `./demo_inference.py`.
Have fun!


## Citing LAVT
```
@inproceedings{yang2022lavt,
  title={LAVT: Language-Aware Vision Transformer for Referring Image Segmentation},
  author={Yang, Zhao and Wang, Jiaqi and Tang, Yansong and Chen, Kai and Zhao, Hengshuang and Torr, Philip HS},
  booktitle={CVPR},
  year={2022}
}
```


## Contributing
We appreciate all contributions.
It helps the project if you could
- report issues you are facing,
- give a :+1: on issues reported by others that are relevant to you,
- answer issues reported by others for which you have found solutions,
- and implement helpful new features or improve the code otherwise with pull requests.

## Acknowledgements
Code in this repository is built upon several public repositories.
Specifically,
* data pre-processing leverages the [refer](https://github.com/lichengunc/refer) repository,
* the backbone model is implemented based on code from [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation),
* the training and testing pipelines are adapted from [RefVOS](https://github.com/miriambellver/refvos),
* and implementation of the BERT model (files in the bert directory) is from [Hugging Face Transformers v3.0.2](https://github.com/huggingface/transformers/tree/v3.0.2)
(we migrated over the relevant code to fix a bug and simplify the installation process).

Some of these repositories in turn adapt code from [OpenMMLab](https://github.com/open-mmlab) and [TorchVision](https://github.com/pytorch/vision).
We'd like to thank the authors/organizations of these repositories for open sourcing their projects.


## License
GNU GPLv3
