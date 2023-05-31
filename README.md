# Probabilistic computation with emerging covariance: towards efficient uncertainty quantification
###  [Paper](https://arxiv.org/abs/2305.19265)
> [**Probabilistic computation with emerging covariance: towards efficient uncertainty quantification**](https://arxiv.org/abs/2305.19265),            
> Hengyuan Ma, Yang Qi, Li Zhang, Wenlian Lu, Jianfeng Feng

This repository provides implementation for training and evaluating moment neural networks based on the Pytorch library.

## Dependencies
* python 3.7
* pytorch: 1.11.0
* torchvision: 0.12.0
* scipy: 1.7.3
* numpy: 1.21.6

## Model summary
In `models.py` module, we provide full-connected MNN (Heaviside2, ReluNet2), mixed MNN (CNN_ReluNet2, CNN_Heaviside2), and the corresponding verision using batch-wise trick (Heaviside, ReluNet, CNN_ReluNet, CNN_Heaviside).

## The architecture of this repository

* `activations`: Heaviside and ReLU moment activation.
* `models`: network architectures of MNN such as Heaviside, ReluNet, CNN_ReluNet, CNN_Heaviside.
* `utils`: useful utilities for analyzing results of MNN.
* `dataset`: preprocessing of datasets.
* `test_model`: examples for training and evaluating MNN.

## Demo
Training moment neural network (MNN) through supervised mean and unsupervised covariance (SMUC) 
```
$ python test_model.py --dataset cifar10 
                       --train_batch 256 
                       --test_batch 128 
                       --opt Adam
                       --lr 0.0001 
                       --epochs 120 
                       --weight_decay 0.0005  
                       --save_model
```
* `dataset` is the name of dataset (mnist or cifar10). 
* `train_batch` is the batch size of training.
* `test_batch` is the batch size of evaulation.
* `opt` is the type of optimizer (adam or sgd).
* `lr` is the learning rate.
* `epochs` is the number of training epoches.
* `weight_decay` is the factor of weight decay.
* `save_model` is the option whether to save the model checkpoint.
