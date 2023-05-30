import os
import numpy as np
import torch
import torch.nn as nn
from models import HeavisideNet, HeavisideNet2, ReluNet, ReluNet2, CNN_HeavisideNet, \
    CNN_HeavisideNet2, CNN_ReluNet, CNN_ReluNet2
from dataset import mnist, cifar10
from utils import entropy_cal, right_wrong_stats_report
from tqdm import tqdm

import argparse
import logging
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str, default=None, help='The path of the results')
    parser.add_argument('--dataset', type=str, default='mnist', help='The name of dataset.')
    parser.add_argument('--loss', type=str, default='mse', help='The type of loss.')
    parser.add_argument('--epochs', type=int, default=50, help='The number of training epochs.')
    parser.add_argument('--report_step', type=int, default=100, help='The step interval for report loss.')
    parser.add_argument('--eval_step', type=int, default=1000, help='The step interval for evaluation.')
    parser.add_argument('--train_batch', type=int, default=128, help='The number of training batch size.')
    parser.add_argument('--test_batch', type=int, default=128, help='The number of testing batch size.')
    parser.add_argument('--load_model_path', type=str, default=None, help='The path of the model to be loaded.')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate.')
    parser.add_argument('--gpu', action='store_true', help='If use gpu.')
    parser.add_argument('--save_model', action='store_true', help='If save the model.')
    parser.add_argument('--scheduler', type=str, default=None, help='the scheduler of learning.')
    parser.add_argument('--opt', type=str, default='adam', help='the optimizer')
    parser.add_argument('--amp', action='store_true', help='If use data augmentation.')
    parser.add_argument('--logname', type=str, default='log', help='The name of the log file.')
    parser.add_argument('--mid_C', action='store_true', help='If calculate the stats for middle layers at evaluation.')
    parser.add_argument('--weight_decay', type=float, default=0., help='The coef of weight decay.')
    parser.add_argument('--no_cov', action='store_true', help='If do not consider the correlations.')

    args = parser.parse_args()
    return args


args = parse_args()

workdir = r'save_models/{}_{}_{}'.format(datetime.now().date(), datetime.now().hour, datetime.now().minute,
                                         datetime.now().second)
os.makedirs(workdir)
save_model_path = r'{}/checkpoint.pt'.format(workdir)
config_path = r'{}/config.txt'.format(workdir)

level = getattr(logging, 'INFO', None)
log_path = '{}/log.txt'.format(workdir)

handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(log_path)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.setLevel(level)

amp = args.amp
loss_type = args.loss
epochs = args.epochs
dataset = args.dataset

train_batch = args.train_batch
test_batch = args.test_batch

if dataset == 'mnist':
    train_loader, test_loader, demo_loader = mnist(train_batch, test_batch, amp=False)
    dim_in = 28 * 28
    dim_out = 10
elif dataset == 'cifar10':
    train_loader, test_loader, demo_loader = cifar10(train_batch, test_batch, amp=amp)
    dim_in = 32 * 32 * 3
    dim_out = 10

else:
    raise EOFError('no such dataset')

save_model = args.save_model

device = 'cuda:0'

if dataset == 'mnist':
    conv_net = False
    dim_s = np.array([dim_in, 392, 196, 96, 48, 24])
    Cs_list = np.array([1, 1, 1, 1, 1]) * 0.2
    C_1 = torch.tensor(0.2)
    #net = HeavisideNet2(dim_s, dim_out).to(device)
    net = HeavisideNet(dim_s, dim_out).to(device)
elif dataset == 'cifar10':
    use_bias = True
    pooltype = 'max'
    conv_net = True
    conv_settings = ((3, 64, 3, 1, 1), (64, 64, 3, 1, 1),
                     (64, 128, 3, 1, 1), (128, 128, 3, 1, 1),
                     (128, 256, 3, 1, 1), (256, 256, 3, 1, 1),
                     (256, 256, 3, 1, 1), (256, 256, 3, 1, 1),
                     (256, 256, 3, 1, 1), (256, 256, 3, 1, 1))

    dim_s = np.array([256, 64, 32])
    Cs_conv = np.ones(len(conv_settings)) * 0.1
    bn_s = [True] * len(conv_settings)
    pool_s = [False, True] * (len(conv_settings) // 2)
    Cs_list = np.array([1, 1]) * 0.2
    C_1 = torch.tensor(0.2)
    net = CNN_HeavisideNet2(dim_s=dim_s, dim_out=dim_out, conv_settings=conv_settings, use_bias=use_bias,
                            pooltype=pooltype, bn_s=bn_s, pool_s=pool_s).to(device)

    if conv_net:
        conv_dims = []
        divide_h_w = 1
        for i in range(len(conv_settings)):
            dim = dim_in // conv_settings[0][0] * conv_settings[i][1]
            if pool_s[i]:
                divide_h_w *= 2
            dim = dim // divide_h_w ** 2
            conv_dims.append(dim)

logging.info(net)

total_params = sum(p.numel() for p in net.parameters())
logging.info('total parameters: {}'.format(total_params))

if args.load_model_path is not None:
    net.load_state_dict(torch.load(args.load_model_path), strict=True)


def eval(net, test_loader, output_cov_activated_mid=False):
    net.eval()
    correct = 0
    total = 0
    right_wrong_flag = []
    layer_num = net.layer_num - 1

    if not conv_net:
        net.conv_num = 0

    if output_cov_activated_mid:
        entropy_layer_s = []
        for l in range(layer_num):
            entropy_layer_s.append([])
    entropy_s = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            x, y = data
            y = y.to(device)

            if output_cov_activated_mid:
                if conv_net:
                    output, cov_activated_mids, output_C = net(x.to(device),
                                                               output_C=True,
                                                               C_in=C_1,
                                                               Cs_list=Cs_list,
                                                               Cs_conv=Cs_conv,
                                                               output_cov_activated_mid=True,
                                                               no_cov=args.no_cov)
                else:
                    output, cov_activated_mids, output_C = net(x.to(device),
                                                               output_C=True,
                                                               C_in=C_1,
                                                               Cs_list=Cs_list,
                                                               output_cov_activated_mid=True,
                                                               no_cov=args.no_cov)
            else:
                output, output_C = net(x.to(device),
                                       output_C=True,
                                       Cs_list=Cs_list,
                                       C_in=C_1,
                                       no_cov=args.no_cov)

            entropy_batch = entropy_cal(output_C)
            for sample_index in range(len(entropy_batch)):
                output_ = output[sample_index, ...]
                y_ = y[sample_index, ...]
                entropy = entropy_batch[sample_index]

                if torch.argmax(output_) == y_:
                    correct += 1
                    right_wrong_flag.append(1)
                else:
                    right_wrong_flag.append(0)
                total += 1

                entropy_s.append(entropy)

                if output_cov_activated_mid:
                    for l in range(layer_num):
                        cov_activated_layer = cov_activated_mids[l][sample_index, ...]
                        if l < net.conv_num:
                            dim_l = conv_dims[l]
                        else:
                            dim_l = dim_s[l - net.conv_num]
                        entropy_layer = entropy_cal(cov_activated_layer, dim_l)
                        entropy_layer_s[l].append(entropy_layer)

    right_wrong_flag = np.array(right_wrong_flag)

    entropy_s = np.array(entropy_s)

    if output_cov_activated_mid:
        entropy_layer_s = np.array(entropy_layer_s)

    if output_cov_activated_mid:
        logging.info('----------------------------------------------------------------')
        logging.info('input entropy: {}'.format(dim_in / 2 * (1 + np.log(2 * np.pi)) + 0.5 * dim_in * np.log(C_1)))
        for l in range(layer_num):
            logging.info('layer {}; entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(l + 1,
                                                                                    *right_wrong_stats_report(
                                                                                        entropy_layer_s[l, :],
                                                                                        right_wrong_flag)))
        logging.info('final entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(
            *right_wrong_stats_report(entropy_s, right_wrong_flag)))

    else:
        logging.info('----------------------------------------------------------------')
        logging.info('input entropy: {}'.format(dim_in / 2 * (1 + np.log(2 * np.pi)) + 0.5 * dim_in * np.log(C_1)))
        logging.info('final entropy; rm: {}, rv: {}, wm: {}, wv: {}'.format(
            *right_wrong_stats_report(entropy_s, right_wrong_flag)))

    logging.info(f'accuracy: {round(correct / total, 4)}')


def train(net, train_loader, epochs, loss_type, mid_C=False):
    steps = 0
    if loss_type == 'mse':
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise NameError('no such optimizer')
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    else:
        scheduler = None

    for epoch in range(epochs):
        net.train()

        for data in train_loader:
            x, y = data
            optimizer.zero_grad()
            if conv_net:
                output = net(x.to(device), C_in=C_1, Cs_list=Cs_list, Cs_conv=Cs_conv, output_C=False,
                             no_cov=args.no_cov)
            else:
                output = net(x.to(device), C_in=C_1, Cs_list=Cs_list, output_C=False,
                             no_cov=args.no_cov)
            if loss_type == 'mse':
                loss = loss_func(output, torch.nn.functional.one_hot(y, num_classes=dim_out).float().to(device))
            else:
                loss = loss_func(output, y.to(device))

            loss.backward()
            optimizer.step()
            steps += 1

            if scheduler is not None:
                scheduler.step()

            if steps % args.report_step == 0:
                logging.info('loss: {} steps: {}'.format(loss, steps))

            if steps > 1 and steps % args.eval_step == 0:
                eval(net, test_loader, output_cov_activated_mid=mid_C)

    if save_model:
        torch.save(net.state_dict(), save_model_path)
    return


train(net, train_loader, epochs=epochs, loss_type=loss_type, mid_C=args.mid_C)
eval(net, test_loader, output_cov_activated_mid=True)
