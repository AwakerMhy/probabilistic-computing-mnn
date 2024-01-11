import numpy as np
import torch
import torch.nn as nn
from models import HeavisideNet, ReluNet, HeavisideNet2, ReluNet2
from uci_reg_dataset import Dataset
from utils import entropy_cal, ll_cal, stats_report
from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='concrete', help='The name of dataset.')
    # housing (boston), concrete, energy, kin8, protein, power, wine, yacht

    parser.add_argument('--heaviside', action='store_true', help='If use heaviside activation.')
    parser.add_argument('--batch_cov', action='store_true', help='If true, use batch-wise covariance trick')

    parser.add_argument('--epochs', type=int, default=500, help='The number of training epochs.')
    parser.add_argument('--train_batch', type=int, default=64, help='The number of training batch size.')
    parser.add_argument('--test_batch', type=int, default=1, help='The number of testing batch size.')
    parser.add_argument('--load_model_path', type=str, default=None, help='The path of the model to be loaded.')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate.')
    parser.add_argument('--eval', action='store_true', help='If evaluate the model in the end.')
    parser.add_argument('--scheduler', type=str, default=None, help='the scheduler of learning.')
    parser.add_argument('--opt', type=str, default='adam', help='the optimizer')
    parser.add_argument('--no_cov', action='store_true', help='Do not consider covariance.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--data_norm', action='store_true', help='If rescale the coordinates of dataset.')
    parser.add_argument('--norm_type', type=str, default='mean_std',
                        help='the type of data normalization, can be mean_std or min_max.')
    parser.add_argument('--test_times', type=int, default=5, help='Times of tests.')

    parser.add_argument('--datadir', type=str, default=None, help='If dir for of the UCI dataset.')

    parser.add_argument('--sigma1', type=float, default=1., help='sigma1.')
    parser.add_argument('--sigma2', type=float, default=1., help='sigma2')

    args = parser.parse_args()
    return args


def eval(net, x_test, y_test, no_cov=False):
    layer_num = net.layer_num - 1

    ignore_nan = False

    entropy_layer_s = []
    for l in range(layer_num):
        entropy_layer_s.append([])

    ll_s = []
    with torch.no_grad():
        for i in tqdm(range(y_test.shape[0])):
            x = x_test[i, :].to(device)
            y = y_test[i, :].to(device)

            output, cov_activated_mids, variance_out = net(x.to(device), output_C=True, C_in=C_1,
                                                           Cs_list=Cs_list, output_cov_activated_mid=True,
                                                           no_cov=no_cov)

            if not args.batch_cov:
                variance_out = variance_out[0]

            variance = variance_out.cpu().numpy()
            ll_s.append(
                ll_cal(output.cpu().numpy().reshape(-1), y.cpu().numpy().reshape(-1), variance.reshape(-1),rescale=y_scale.numpy()).reshape(-1)[
                    0])

            for l in range(layer_num):
                cov_activated_layer = cov_activated_mids[l]
                if not args.batch_cov:
                    cov_activated_layer = cov_activated_layer[0]
                dim_l = dim_s[l]
                entropy_layer = entropy_cal(cov_activated_layer, dim_l)
                entropy_layer_s[l].append(entropy_layer)

    entropy_layer_s = np.array(entropy_layer_s)

    print('----------------------------------------------------------------')
    print('input entropy: {}'.format(dim_in / 2 * (1 + np.log(2 * np.pi)) + 0.5 * dim_in * np.log(C_1)))
    for l in range(layer_num):
        print('layer {}; entropy; mean: {}, variance: {}.'.format(
            l + 1, *stats_report(entropy_layer_s[l, :], ignore_nan=ignore_nan)))
    print('----------------------------------------------------------------')
    ll_mean, ll_std = stats_report(ll_s, ignore_nan=False)
    print('ll, mean: {}, std: {}'.format(ll_mean, ll_std))
    return ll_mean


args = parse_args()

no_cov = args.no_cov
epochs = args.epochs
dataset = args.dataset
train_batch = args.train_batch
test_batch = args.test_batch
path_dir = args.datadir

special_list = ['kin8', 'power']

if dataset not in special_list:
    data = Dataset(dataset, path=path_dir)
else:
    data = np.load(r'{}/{}/data.npy'.format(path_dir, dataset))

device = 'cuda:0'

ll_mean_s = []

if dataset in special_list:
    data_num = data.shape[0]
    random_index_s = []
    for i in range(10):
        random_index_s.append(np.random.permutation(data_num)[i * int(data_num / 10):(i + 1) * int(data_num / 10)])

for n in tqdm(range(args.test_times)):
    print('----------------------------------------------------------')
    print('test times: {}'.format(n))

    if dataset not in special_list:
        x_train, y_train, x_test, y_test = data.get_split(split=int(np.random.choice(np.arange(10))))
    else:
        index_ = random_index_s[int(np.random.choice(np.arange(10)))]
        rest_index = np.setdiff1d(np.arange(data.shape[0]), index_)
        x_train = data[rest_index, :-1]
        y_train = data[rest_index, -1].reshape(-1, 1)
        x_test = data[index_, :-1]
        y_test = data[index_, -1].reshape(-1, 1)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    if args.data_norm:
        if args.norm_type == 'mean_std':
            x_train_std = x_train.std(dim=0)
            y_train_std = y_train.std(dim=0)

            x_train_mean = x_train.mean(dim=0)
            y_train_mean = y_train.mean(dim=0)

            x_train = (x_train - x_train_mean) / x_train_std
            y_train = (y_train - y_train_mean) / y_train_std
            x_test = (x_test - x_train_mean) / x_train_std
            y_test = (y_test - y_train_mean) / y_train_std
            y_scale = y_train_std
        elif args.norm_type == 'min_max':
            x_train_max = x_train.max(dim=0)[0]
            y_train_max = y_train.max(dim=0)[0]

            x_train_min = x_train.min(dim=0)[0]
            y_train_min = y_train.min(dim=0)[0]

            x_train = (x_train - x_train_min) / (x_train_max - x_train_min)
            y_train = (y_train - y_train_min) / (y_train_max - y_train_min)
            x_test = (x_test - x_train_min) / (x_train_max - x_train_min)
            y_test = (y_test - y_train_min) / (y_train_max - y_train_min)
            y_scale = (y_train_max - y_train_min)
        else:
            raise NameError('No such normalization type.')
    else:
        y_scale = 1

    train_num = x_train.shape[0]

    dim_in = x_train.shape[1]
    dim_s = np.array([dim_in, 50])

    C_1 = torch.tensor(1) * args.sigma1 ** 2
    Cs_list = np.array([1]) * args.sigma2 ** 2

    dim_out = 1

    if args.heaviside:
        if args.batch_cov:
            net = HeavisideNet(dim_s, dim_out).to(device)
        else:
            net = HeavisideNet2(dim_s, dim_out).to(device)
    else:
        if args.batch_cov:
            net = ReluNet(dim_s, dim_out).to(device)
        else:
            net = ReluNet2(dim_s, dim_out).to(device)

    loss_func = nn.MSELoss()

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        raise NameError('no such optimizer')
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    elif args.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    else:
        scheduler = None

    steps = 0
    for epoch in range(epochs):
        random_index = torch.randperm(train_num)
        for i in range(train_num // train_batch):
            data_index = random_index[train_batch * i:train_batch * (i + 1)]
            x = x_train[data_index, :]
            y = y_train[data_index, :]
            optimizer.zero_grad()
            output = net(x.to(device), C_in=C_1, Cs_list=Cs_list, output_C=False, no_cov=no_cov)

            loss = loss_func(output, y.to(device))
            loss.backward()
            optimizer.step()
            steps += 1

            if scheduler is not None:
                scheduler.step()

            if steps % 2000 == 0:
                print('loss: {} steps: {}'.format(loss, steps))

    ll_mean = eval(net, x_test, y_test, no_cov=no_cov)
    ll_mean_s.append(ll_mean)

print('final ll mean:{:.2f}, std: {:.2f}'.format(*stats_report(ll_mean_s, ignore_nan=False)))
