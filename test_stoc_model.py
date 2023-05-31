import matplotlib.pyplot as plt
import numpy as np
import torch
from stoc_models import sto_Net, sto_Net2
from models import HeavisideNet2, ReluNet2
from dataset import mnist
from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch', type=int, default=1, help='The number of testing batch size.')
    parser.add_argument('--load_model_path', type=str, default=None, help='The path of the model to be loaded.')
    parser.add_argument('--times', type=int, default=10, help='The number of MC samples.')
    parser.add_argument('--deter_eval',action='store_true',help='If set all the second moments as zero when inference.')
    parser.add_argument('--dt',type=float,default=0.5,help='The length of step.')
    parser.add_argument('--net_type',type=str,default='relu',help='The type of activations.')
    args = parser.parse_args()
    return args

args = parse_args()

test_batch = args.test_batch
train_loader, test_loader, demo_loader = mnist(1, test_batch, amp=False)
dim_in = 28 * 28
dim_out = 10
device = 'cuda:0'


dim_s = np.array([dim_in, 196, 48])
Cs_list = np.array([1, 1]) * 0.4
C_1 = torch.tensor(1) * 0.4
net_type = args.net_type # 'heav' # relu

net = sto_Net2(net_type, dim_s, dim_out).to(device)
net_steady = sto_Net(net_type, dim_s, dim_out).to(device)


if net_type == 'heav':
    net_deter = HeavisideNet2(dim_s, dim_out).to(device)
elif net_type == 'relu':
    net_deter = ReluNet2(dim_s, dim_out).to(device)
else:
    raise NameError('No such network type')

if args.load_model_path is not None:
    net.load_state_dict(torch.load(args.load_model_path), strict=True)
    net_deter.load_state_dict(torch.load(args.load_model_path), strict=True)
    net_steady.load_state_dict(torch.load(args.load_model_path), strict=True)
dt = args.dt
net.eval()


vmax = 10
show_length = 500

with torch.no_grad():
    for data in tqdm(test_loader):

        state_records = []
        state_records_steady = []

        for i in range(len(dim_s)):
            state_records.append([])
            state_records_steady.append([])
        outputs = []
        outputs_steady = []

        x, y = data
        y = y.to(device)
        if args.deter_eval:
            zero_moment_flag = 0
        else:
            zero_moment_flag = 1

        zero_moment_flag_ = 0

        states = net(x.to(device), Cs_list=Cs_list*zero_moment_flag_, C_in=C_1*zero_moment_flag_)
        states_steady = net_steady(x.to(device), Cs_list=Cs_list*zero_moment_flag, C_in=C_1*zero_moment_flag)
        output, output_C = net_deter(x.to(device), output_C=True, Cs_list=Cs_list, C_in=C_1)

        for state in states:
            state *= 0

        outputs.append(states[-1].cpu().numpy())
        outputs_steady.append(states_steady[-1].cpu().numpy())

        for i in range(len(dim_s)):
            state_records[i].append(states[i].cpu().numpy())
            state_records_steady[i].append(states_steady[i].cpu().numpy())


        for t in range(args.times):
            states = net.run(dt, x.to(device), states, Cs_list=Cs_list * zero_moment_flag, C_in=C_1 * zero_moment_flag)
            states_steady = net_steady(x.to(device), Cs_list=Cs_list * zero_moment_flag, C_in=C_1 * zero_moment_flag)

            outputs.append(states[-1].cpu().numpy())
            outputs_steady.append(states_steady[-1].cpu().numpy())
            for i in range(len(dim_s)):
                state_records[i].append(states[i].cpu().numpy())
                state_records_steady[i].append(states_steady[i].cpu().numpy())
        for i in range(len(dim_s)):
            state_records[i] = np.concatenate(state_records[i])
            state_records_steady[i] = np.concatenate(state_records_steady[i])

        outputs = np.concatenate(outputs[5000:])
        outputs_steady = np.concatenate(outputs_steady[5000:])

        covariance_est = np.cov(outputs[:,:], rowvar=False)
        mean_est = np.mean(outputs.T,axis=1)

        covariance_est_steady = np.cov(outputs_steady[:,:], rowvar=False)
        variance_est_steady = np.var(outputs_steady[:,:].T,axis=1)
        mean_est_steady = np.mean(outputs_steady .T,axis=1)

        fontsize = 15

        fig, axs = plt.subplots(nrows=len(dim_s)+1, ncols=1, sharex=True, sharey=False,figsize=[6,7])
        plt.subplots_adjust(wspace=0.0, hspace=0.05)
        for i in range(len(dim_s)):
            axs[i].imshow(state_records[i][:show_length].T,aspect=80/state_records[i].shape[1])
            axs[i].set_yticks([])
            axs[i].set_xticks([])
            axs[i].set_ylabel('layer {}'.format(i+1),fontsize=fontsize)

        axs[-1].imshow(outputs[:show_length].T,aspect=80/outputs.shape[1])
        axs[-1].set_ylabel('pred label',fontsize=fontsize)
        axs[-1].set_xlabel('times',fontsize=fontsize)
        axs[-1].set_yticks([0,2,4,6,8],[0,2,4,6,8],rotation=20,fontsize=fontsize-2)
        axs[-1].set_xticks([0,show_length//2,show_length],fontsize=fontsize)


        fig2 = plt.figure()
        plt.imshow(x[0,0],cmap='Greys_r')
        plt.xticks([])
        plt.yticks([])

        plt.axis('off')

        fig3, axes3 = plt.subplots(figsize=[5,5],nrows=1, ncols=2)

        img1 = axes3[0].imshow(output_C[0,...].cpu(),cmap='coolwarm',vmax=vmax,vmin=-vmax)
        axes3[0].set_title('MNN',fontsize=fontsize)
        axes3[0].set_xticks([0,2,4,6,8],[0,2,4,6,8],fontsize=fontsize-2)
        axes3[0].set_yticks([0, 2, 4, 6, 8], [0, 2, 4, 6, 8], fontsize=fontsize - 2)
        axes3[1].set_title('SDE',fontsize=fontsize)
        axes3[1].imshow(covariance_est, cmap='coolwarm', vmax=vmax, vmin=-vmax)
        axes3[1].set_xticks([0,2,4,6,8],[0,2,4,6,8],fontsize=fontsize-2)
        axes3[1].set_yticks([0, 2, 4, 6, 8], [0, 2, 4, 6, 8], fontsize=fontsize - 2)
        fig3.colorbar(img1, ax=axes3.ravel().tolist())

        fig4 = plt.figure(figsize=[6,4.5])
        plt.subplots_adjust(wspace=0.0, hspace=0.2)
        plt.subplot(2, 1, 1)
        plt.bar(np.arange(mean_est.shape[0]) * 5, output[0,:].cpu(), alpha=0.6, color='blue')
        plt.bar(np.arange(mean_est.shape[0]) * 5 + 1, mean_est, alpha=0.6, color='green')
        plt.xticks([])

        plt.ylabel('mean',fontsize=fontsize)
        plt.legend(['MNN', 'SDE'],fontsize=fontsize)

        plt.subplot(2, 1, 2)
        plt.bar(np.arange(mean_est.shape[0]) * 5,torch.diag(output_C[0,...].cpu()), alpha=0.6, color='blue')
        plt.bar(np.arange(mean_est.shape[0]) * 5 + 1, np.diag(covariance_est), alpha=0.6, color='green')
        plt.xticks(np.arange(mean_est.shape[0]) * 5 + 0.5,np.arange(mean_est.shape[0]),fontsize=fontsize)

        plt.ylabel('variance',fontsize=fontsize)
        plt.legend(['MNN', 'SDE'],fontsize=fontsize)
        plt.xlabel('predicted label',fontsize=fontsize)
        plt.show()

