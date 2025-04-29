# Crack Detection in Paintings by Vincent van Gogh

This repository contains Python code for training and evaluating generative machine learning models to detect cracks in paintings by Vincent van Gogh.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project aims to develop and evaluate generative machine learning models for detecting cracks in paintings by Vincent van Gogh. The models are trained on a dataset of high-resolution images of reproductions of Van Gogh's paintings, with the goal of identifying and highlighting cracks with lifted edges. More specifically, generative models are used for reconstruction-based out of distribution detection, therefore this repository features multiple generative models for that purpose.

This repository is forked from [HERE](https://github.com/ashleve/lightning-hydra-template), a convenient template for machine learning related experiments. Any documentation regarding training, validating and testing machine learning models, can be found in there. The READme of this repository is also included in the `docs` folder

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/levimohle/ml-crack-detection-van-gogh.git
cd ml-crack-detection-van-gogh
pip install -r requirements.yaml
```

## Dataset

The dataset contains RGB images and height maps of 512x512 px, from reproductions of Van Gogh paintings. Below some examples are included.

<p align="center">
  <img src="imgs/rgb_example.png" width="40%" alt='RGB images'>
  <img src="imgs/height_example.png" width="40%" alt='Height images'>
</p>

## Model Training

```bash
# Example command to train the model
python train.py experiment=impasto_cddpm_final trainer=ddp trainer.devices=2

# Example command to evaluate the model
python eval.py +experiment=impasto_cddpm_final trainer=gpu ckpt_path=/best_model.ckpt
```

## Evaluation

## Acknowledgements

## Citation
If you publish work that uses the work presented in this repository, please cite ml-crack-detection-van-gogh as follows:

```bibtex
@misc{ml-crack-detection-van-gogh,
author = {Levi Möhle},
title = {Reconstruction-based out of distribution detection for paintings by Vincent van Gogh},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/levimohle/ml-crack-detection-van-gogh}},
year = {2025},
}
```

For citing the paper, please cite as follows:

```bibtex
@misc{crackdetectionvg_2025,
author = {L.S. Möhle},
title = {Crack detection in paintings by Vincent van Gogh},
publisher = {Eindhoven University of Technology},
journal = {Research portal Eindhoven University of Technology},
howpublished = {\url{}},
year = {2025},
}
```
## License

MIT License

Copyright (c) [2025] [Levi Mohle]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
