# Geometric-fakenews-detection

Repository for the project on detecting fakenews using geometric deep learning for the Geometric data analysis MVA course.

We provide an implementation of the paper [Fake News Detection on Social Media using Geometric Deep Learning](https://arxiv.org/pdf/1902.06673.pdf). As well as some variant of their model.

The dataset used is the UPFD dataset made available by the authors of the paper [User Preference-aware Fake News Detection](https://arxiv.org/pdf/2104.12259.pdf) based on the [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) dataset.

## Requirements

To install the necessary requirements, run the following commands:

``` bash
pip install -r requirements.txt
```

## Usage

An example of training the model on the gossipcop dataset using the profile features is given by:

``` bash
python train.py --dataset gossipcop --features profile --epochs 600 --batch_size 128 --model_name model.pt
```

Find more information about the arguments by running:

``` bash
python train.py --help
```

The model will be saved in the `models` folder.

## Results

See the [report](report.pdf) for more details.
