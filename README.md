# Probabilistic Computation with Emerging Covariance: Towards Efficient Uncertainty Quantification
###  [Paper](https://arxiv.org/abs/2305.19265)
> [**Probabilistic computation with emerging covariance: towards efficient uncertainty quantification**](https://arxiv.org/abs/2305.19265),            
> Hengyuan Ma, Yang Qi, Li Zhang, Wenlian Lu, Jianfeng Feng

This repository provides implementation for training and evaluating moment neural networks (MNNs) trained through
the Supervised Mean and Unsupervised Covariance (SMUC) approach based on the Pytorch library.


## Abstract
Building robust, interpretable, and secure artificial intelligence system requires some degree of quantifying and representing uncertainty 
via a probabilistic perspective, as it allows 
to mimic human cognitive abilities. 
However, probabilistic computation presents significant challenges due to its inherent complexity.
In this paper, we develop an efficient and interpretable probabilistic computation framework by truncating the probabilistic 
representation up to its first two moments, i.e., mean and covariance.
We instantiate the framework by training a deterministic surrogate of a stochastic network that learns the complex probabilistic 
representation via combinations of simple activations, 
encapsulating the non-linearities coupling of the mean and covariance.
We show that when the mean is supervised for optimizing the task objective, the unsupervised covariance spontaneously emerging from 
the non-linear coupling with the mean faithfully 
captures the uncertainty associated with model predictions.
Our research highlights the inherent computability and simplicity of probabilistic computation, enabling its wider application in large-scale settings.


## Model summary
In `models.py` module, we provide full-connected MNN (Heaviside2, ReluNet2), mixed MNN (CNN_ReluNet2, CNN_Heaviside2), 
and the corresponding verision using batch-wise trick (Heaviside, ReluNet, CNN_ReluNet, CNN_Heaviside).


## The architecture of this repository

* `activations.py`: Heaviside and ReLU moment activation.
* `models.py`: network architectures of MNN such as Heaviside, ReluNet, CNN_ReluNet, CNN_Heaviside.
* `stoc_models.py`: full-connected feed-forward stochastic neural network.
* `utils.py`: useful utilities for analyzing results of MNN.
* `dataset.py`: preprocessing of datasets.
* `test_classification.py`: examples for training and evaluating MNN for classification tasks.
* `test_regression.py`: examples for training and evaluating MNN for regression tasks.
* `test_ood.py`: demo for out-of-detection detection using MNN.
* `test_attack.py`: demo for gradient-based adversarial attack defending and awareness using MNN.
* `test_stoc_models.py`: demo for simulating stochastic network using parameters of MNN.
* `data`: directory for storing datasets such as UCI regression dataset and notMNIST dataset.
* `checkpoints`: directory for storing some pretrained checkpoints.

## Dependencies
* python 3.7
* pytorch: 1.11.0
* torchvision: 0.12.0
* scipy: 1.7.3
* numpy: 1.21.6
* matplotlib: 3.5.1


## Demo

### Classification task
Code for training a Moment Neural Network (MNN) using the Supervised Mean and Unsupervised Covariance (SMUC) approach
for image classification tasks on MNIST, FashionMNIST, CIFAR-10, and CIFAR-100.
To provide a demonstration, we have constructed different architectures for each dataset, and feel free 
to modify them according to your specific needs.
For the MNIST dataset, we have included an example of a fully-connected network.
For the FashionMNIST dataset, we have provided an example of a modified LeNet5 network.
For the CIFAR-10 dataset, we have included an example of a modified VGG13 network. 
For the CIFAR-100 dataset, we have included an example of a modified VGG13 network with large channels number. 
```
$ python test_classification.py --dataset cifar10 
                                --heaviside
                                --batch_cov
                                --amp
                                --train_batch 256 
                                --test_batch 128 
                                --opt adam
                                --lr 0.0001 
                                --epochs 120 
                                --weight_decay 0.0005  
                                --save_model
```
* `dataset` is the name of dataset (mnist, fashionmnist, cifar10, or cifar100). 
* `amp` is the option for whether to use data amplification
* `heaviside` is the option for whether to use Heaviside moment activations, if not true, ReLY moment activations are applied.
* `batch_cov` is the option for whether to use batch-wise covariance trick during training (see Supplementary Information of the paper for details).
* `train_batch` is the batch size of training.
* `test_batch` is the batch size of evaulation.
* `opt` is the type of optimizer (adam or sgd).
* `lr` is the learning rate.
* `epochs` is the number of training epoches.
* `weight_decay` is the factor of weight decay.
* `save_model` is the option for whether to save the model checkpoint.

### Regression task
Code for training a Moment Neural Network (MNN) using the Supervised Mean and Unsupervised Covariance (SMUC) approach
for regression tasks on the UCI regression datasets. Before running the following experiment,
please download the `data/notMNIST_small.zip` and extract its contents.

```
$ python test_regression.py --datadir data\uci_datasets
                            --dataset concrete 
                            --heaviside
                            --data_norm
                            --norm_type min_max
                            --batch_cov
                            --train_batch 64 
                            --opt sgd
                            --lr 0.0001 
                            --epochs 1000 
                            --save_model
                            --sigma1 0.5
                            --sigma2 0.5
                            --test_times 20
```
* `datadir` is the directory of dataset.
* `dataset` is the name of dataset (housing, concrete, energy, kin8, protein, power, wine, yacht).
* `heaviside` is the option for whether to use Heaviside moment activations, if not true, ReLU moment activations are applied.
* `data_norm`is the option for whether to normalize the dataset before training.
* `norm_type`is the typy of data normalization (min-max, mean_std)
* `batch_cov` is the option for whether to use batch-wise covariance trick during training (see Supplementary Information of the paper for details).
* `train_batch` is the batch size of training.
* `opt` is the type of optimizer (adam or sgd).
* `lr` is the learning rate.
* `epochs` is the number of training epoches.
* `sigma1` is the setting of the hyperparameter $\sigma_1$.
* `sigma2` is the setting of the hyperparameter $\sigma_2$.
* `test_times` is the times of the experiments.

### Out-of-distribution detection
Code for out-of-distribution (OOD) detection using MNN. Before running the following experiment, 
please download the pretrained checkpoint `checkpoints/mnist_ood_demo.pt` and the NotMNIST dataset
`data/notMNIST_small.zip` and extract it in the `data` directory.
```
python test_ood.py  
```


### Gradient-based adversarial attack defense and awareness
Code for adversarial attack defense and awareness using Heaviside MNN. 
the FGSM We apply fast gradient sign method (FGSM)
to generate adversarial samples. 
If set the $\sigma_1,\sigma_2$ as zero, FGSM has no effect to the model, otherwise, the model is awareness to the FGSM,
and the output entropy increases.
Here we demonstrate two examples for MNIST and CIFAR-10 dataset respectively.
Before running the following experiment, 
please download the pretrained checkpoint `checkpoints/mnist_ood_demo.pt` for the
MNIST case and `checkpoints/cifar10_attack_demo.pt` for the CIFAR-10 case. 
```
python test_attack.py  --dataset mnist
                       --defending
```
* `dataset` is the name of dataset (mnist, cifar10).
* `defending` is the option for whether to set the $\sigma_1,\sigma_2$ as zero to defend the adversarial attack.

### Numerical simulation on the stochastic network
Code for simulating the feed-forward neural network constructed by the parameters trained on MNN. Before running the following experiment, 
please download the pretrained checkpoint `checkpoints/mnist_heav_mlp.pt` for the
Heaviside case and `checkpoints/mnist_relu_mlp.pt` for the ReLU case.
```
python test_stoc_model.py --net_type  heav
                          --load_model_path checkpoints\mnist_heav_mlp.pt
                          --times 5500
                          --dt 0.1
```
* `net_type` is the type of the moment activation used network of dataset (heav, relu).
* `load_model_path` is the path storing the pretrained checkpoint of MNN.
* `times` is the number of the simulation time steps.
* `dt` is the size of the time step.
