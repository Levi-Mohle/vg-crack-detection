# Crack Detection in Paintings by Vincent van Gogh

This repository contains Python code for training and evaluating generative machine learning models to detect cracks in paintings by Vincent van Gogh.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- License
- [Acknowledgements](#acknowledgements)

## Introduction

This project aims to develop and evaluate generative machine learning models for detecting cracks in paintings by Vincent van Gogh. The models are trained on a dataset of high-resolution images of reproductions of Van Gogh's paintings, with the goal of identifying and highlighting cracks with lifted edges.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/levimohle/ml-crack-detection-van-gogh.git
cd ml-crack-detection-van-gogh
pip install -r requirements.yaml
```

## Dataset


## Model Training

```bash
# Example command to train the model
python train.py experiment=impasto_cddpm_final trainer=ddp trainer.devices=2

# Example command to evaluate the model
python eval.py +experiment=impasto_cddpm_final trainer=gpu ckpt_path=/best_model.ckpt
```

## Evaluation

## Acknowledgements
