## Table of contents
* [Introduction](#Intro)
* [Installation](#Installation)
* [Start](#start)

## Intro
The scope of this project was to see if I was able to translate English to Sicilian dialect using Transformer models. It was particularly difficult as the accessability to these datasets are quite hard to find. Luckily, the OPUS - Corpora provided datasets that were decent enought to create this project. This project required lots of computational power and was finetuned off of the pretrained T5 model provided from HuggingFace. The main difficulty is that Transformer models are not trained off of the Sicilian dialect, so finetuning the T5 model did not yield the desired results. Perhaps in future updates, I will be able to achieve better performance.

## Installation
Project is created with:
* Python 3.11.0

```
$ pip install pytorch
```

## start
To run this project, run the main file

```
$ python src/main.py

```

	![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
