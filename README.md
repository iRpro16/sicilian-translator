# Sicilian Translator

## Table of contents
* [Introduction](#Intro)
* [Installation](#Installation)
* [Start](#Start)
* [Website](Wwebsite)

## Intro
The scope of this project was to see if I was able to translate English to Sicilian dialect using Transformer models. It was particularly difficult as the accessability to these datasets are quite hard to find. Luckily, the OPUS - Corpora provided datasets that were decent enought to create this project. This project required lots of computational power and was finetuned off of the pretrained T5 model provided from HuggingFace. 

The main difficulty was that Transformer models are not trained off of the Sicilian dialect, so finetuning the T5 model did not yield the desired results. Perhaps in future updates, I will be able to achieve better performance. I believe that by first training a model and then finetuning may allow for better results. Creating the tokenizer also proved to be a bit more difficult, as the dialect has many grammatical rules. For this large a dataset, my 4070 RTX GPu did not suffice. So, I utilized the A100 GPU provided by Lambda Labs. 

## Installation
Project is created with:
* Python 3.11.0

```
$ pip install -r requirements.txt
```

## Start
To run this project, run the main file. This will begin to finetune the model.

```
$ python src/main.py

```

## Website
Here's the link: https://sicilian-translator-qvmdhfpokewgzzihbhf9ds.streamlit.app/

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
