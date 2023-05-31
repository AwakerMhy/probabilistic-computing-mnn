import numpy as np
import torch
from models import HeavisideNet2, CNN_HeavisideNet
from dataset import mnist, cifar10
from utils import entropy_cal, stats_report

from tqdm import tqdm
import torch.nn.functional as F
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='The name of dataset.')
    parser.add_argument('--defending', action='store_true', help='If true, set the sigma1 and sigma2 as zero for defending attacks.')
    args = parser.parse_args()
    return args

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

def test_fgsm_attack(model, device, attact_loader, epsilon, C_in_attack=0.00, C_in_attack2=0.00):
    model.eval()
    num_correct = 0
    num_attacked = 0

    entropy_s = []

    for data, target in tqdm(attact_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        Cs_list_ = np.copy(Cs_list)
        Cs_list_[0] = 0

        if conv_net:
            output, output_C = model(data, C_in=C_1 * 0 + C_in_attack, Cs_list=Cs_list_, Cs_conv=Cs_conv,
                                   output_C=True)
        else:
            output, output_C = model(data, output_C=True, Cs_list=Cs_list_, C_in=C_1 * 0 + C_in_attack2)
        pred = output.argmax(dim=1, keepdim=True)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        if conv_net:
            output, output_C = model(perturbed_data, C_in=C_1 * 0 + C_in_attack, Cs_list=Cs_list_, Cs_conv=Cs_conv,
                                   output_C=True)
        else:
            output, output_C = model(perturbed_data, output_C=True, Cs_list=Cs_list_, C_in=C_1 * 0 + C_in_attack2)
        final_pred = output.argmax(dim=1, keepdim=True)

        entropy_after = entropy_cal(output_C, dim_out)
        entropy_s.append(entropy_after)

        num_correct += final_pred.eq(target.view_as(pred)).sum().item()
        num_attacked += target.shape[0]
    accuracy = num_correct / num_attacked
    return accuracy, entropy_s

args = parse_args()

device = 'cuda:0'
dataset = args.dataset  #cifar10 or mnist
amp = True

if dataset == 'mnist':
    train_loader, test_loader, attact_loader = mnist(128, 1, amp=False)
    dim_in = 28 * 28

elif dataset == 'cifar10':
    train_loader, test_loader, attact_loader = cifar10(128, 1, amp=amp)
    dim_in = 32 * 32 * 3
else:
    raise EOFError('no such dataset')


if dataset == 'cifar10':
    conv_net = True
    save_model_path = r'checkpoints/cifar10_attack_demo.pt'
    use_bias = True
    pooltype = 'max'

    conv_settings = ((3, 64, 3, 1, 1), (64, 64, 3, 1, 1),
                     (64, 128, 3, 1, 1), (128, 128, 3, 1, 1),
                     (128, 256, 3, 1, 1), (256, 256, 3, 1, 1),
                     (256, 512, 3, 1, 1), (512, 512, 3, 1, 1),
                     (512, 512, 3, 1, 1), (512, 512, 3, 1, 1))
    dim_s = np.array([512, 128])
    dim_out = 10
    Cs_conv = np.ones(len(conv_settings)) * 0.1
    bn_s = [True] * len(conv_settings)
    pool_s = [False, True] * (len(conv_settings) // 2)
    Cs_list = np.array([1]) * 0.1
    C_1 = torch.tensor(0.5)
    net = CNN_HeavisideNet(dim_s=dim_s, dim_out=10, conv_settings=conv_settings, use_bias=use_bias,
                      pooltype=pooltype, bn_s=bn_s, pool_s=pool_s).to(device)

elif dataset == 'mnist':
    conv_net = False
    save_model_path = r'checkpoints/mnist_attack_demo.pt'

    Cs_list = np.array([1, 1, 1, 1, 1]) * 0.1
    C_1 = torch.tensor(0.5)
    dim_s = np.array([dim_in, 392, 196, 96, 48, 24])
    dim_out = 10
    net = HeavisideNet2(dim_s, dim_out).to(device)

if args.defending:
    C_in_attack = 0.
    C_in_attack2 = 0.
else:
    C_in_attack = 0.2
    C_in_attack2 = 0.1

net.eval()

net.load_state_dict(torch.load(save_model_path), strict=True)
accuracies = []
examples = []


epsilons = np.array([0, .05, .1, .15, .2, .25, .3])

entropy_s_s = []
for eps in epsilons:
    acc, entropy_s = test_fgsm_attack(net, device, attact_loader, eps, C_in_attack,C_in_attack2)
    accuracies.append(acc)
    entropy_s_s.append(entropy_s)
    print(
        'epsilon: {}, acc: {}, entropy mean: {}, std: {}'.format(eps, acc, *stats_report(entropy_s, ignore_nan=False)))
print(accuracies)
print(epsilons)
