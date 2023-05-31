import numpy as np
import torch
from models import ReluNet2
from dataset import mnist
import torchvision.transforms as transforms
from utils import entropy_cal, right_wrong_stats_report
from tqdm import tqdm


def eval(net, test_loader, output_cov_activated_mid=False):
    net.eval()
    correct = 0
    total = 0

    right_wrong_flag = []
    layer_num = net.layer_num - 1

    if output_cov_activated_mid:
        entropy_layer_s = []
        variance_layer_s = []
        ef_layer_s = []
        for l in range(layer_num):
            entropy_layer_s.append([])
            variance_layer_s.append([])
            ef_layer_s.append([])

    entropy_s = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y = data
            y = y.to(device)
            if output_cov_activated_mid:
                output, cov_activated_mids, output_C = net(x.to(device), output_C=True, C_in=C_1,
                                                           Cs_list=Cs_list,
                                                           output_cov_activated_mid=True)
            else:
                output, output_C = net(x.to(device), output_C=True, Cs_list=Cs_list, C_in=C_1)

            entropy = entropy_cal(output_C, dim_out)

            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                    right_wrong_flag.append(1)
                else:
                    right_wrong_flag.append(0)
                total += 1

                entropy_s.append(entropy)
                if output_cov_activated_mid:
                    for l in range(layer_num):
                        cov_activated_layer = cov_activated_mids[l]
                        dim_l = dim_s[l]
                        entropy_layer = entropy_cal(cov_activated_layer, dim_l)
                        entropy_layer_s[l].append(entropy_layer)


    right_wrong_flag = np.array(right_wrong_flag)

    entropy_s = np.array(entropy_s)

    if output_cov_activated_mid:
        entropy_layer_s = np.array(entropy_layer_s)

    if output_cov_activated_mid:
        print('----------------------------------------------------------------')
        print('input entropy: {}'.format(dim_in / 2 * (1 + np.log(2 * np.pi)) + 0.5 * dim_in * np.log(C_1)))
        for l in range(layer_num):
            print('layer {}; entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(l + 1,
                                                                             *right_wrong_stats_report(
                                                                                 entropy_layer_s[l, :],
                                                                                 right_wrong_flag)))
        print('final; entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(
            *right_wrong_stats_report(entropy_s, right_wrong_flag)))
        print('----------------------------------------------------------------')
    else:
        print('----------------------------------------------------------------')
        print('input entropy: {}'.format(dim_in / 2 * (1 + np.log(2 * np.pi)) + 0.5 * dim_in * np.log(C_1)))
        print('final; entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(
            *right_wrong_stats_report(entropy_s, right_wrong_flag)))
        print('----------------------------------------------------------------')


    print(f'accuracy: {round(correct / total, 4)}')
    net.train()

    return correct / total, entropy_s, right_wrong_flag


device = 'cuda:0'
save_model_path = r'checkpoints/mnist_ood_demo.pt'

Cs_list = np.array([1, 1, 1, 1, 1]) * 0.2

C_1 = torch.tensor(0.2)

dim_in = 28 * 28
dim_s = np.array([dim_in, 392, 196, 96, 48, 24])
dim_out = 10
net = ReluNet2(dim_s, dim_out).to(device)


net.load_state_dict(torch.load(save_model_path), strict=True)

import os
from PIL import Image


datadir = r'data/notMNIST_small/notMNIST_small'
dir_names = os.listdir(datadir)

train_loader, test_loader, attact_loader = mnist(128, 1, amp=False)
dim_in = 28 * 28

def ood():
    entropy_s = []
    sample_num = 0

    with torch.no_grad():
        acc, entropy_s_test, right_wrong_flag_test = eval(net, test_loader, output_cov_activated_mid=False)

        print('acc:{}'.format(acc))

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)), ])
        for dir in tqdm(dir_names):
            img_names = os.listdir(r'{}/{}'.format(datadir, dir))
            for img_name in img_names:
                try:
                    img = Image.open(r'{}/{}/{}'.format(datadir, dir, img_name))

                    sample_num += 1
                    input = transform(np.asarray(img)).float().unsqueeze(0)
                    output, output_C = net(input.to(device), output_C=True, Cs_list=Cs_list, C_in=C_1)

                    entropy_batch = entropy_cal(output_C)
                    for sample_index in range(output_C.shape[0]):
                        entropy = entropy_batch[sample_index]
                        entropy_s.append(entropy)

                    if sample_num % 1000 == 0:
                        print('num: {} ---------------------'.format(sample_num))
                        print('mean entropy: {}, var entropy: {}'.format(np.nanmean(entropy_s), np.nanvar(entropy_s)))

                except:
                    print('no png file')
                    continue
ood()