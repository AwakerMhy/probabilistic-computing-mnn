import numpy as np
import torch
import torch.nn as nn


# steady-state distribution
class sto_Net(nn.Module):
    def __init__(self, activation='heav', dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(sto_Net, self).__init__()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.dim_out = dim_out
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        if activation == 'heav':
            self.act = lambda x: (x > 0).float()
        elif activation == 'relu':
            self.act = nn.ReLU()

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1])):
        assert Cs_list.shape[0] == len(self.linear_sequence)
        # the layer before mnn
        result_states = []
        x = img.view(-1, self.dim_in) + torch.randn_like(img.view(-1, self.dim_in)) * np.sqrt(C_in)
        # first layer
        if len(self.linear_sequence) > 1:
            x = self.act(self.linear_sequence[0](x))
            x = x + torch.randn_like(x) * np.sqrt(Cs_list[0])
            result_states.append(x)
        # middle layers
        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x = self.act(linear_func(x))
            x = x + torch.randn_like(x) * np.sqrt(Cs_list[i + 1])
            result_states.append(x)
        # final layer
        x = self.act(self.linear_sequence[-1](x))
        x = x + torch.randn_like(x) * np.sqrt(Cs_list[-1])
        result_states.append(x)
        x = self.final(x)
        result_states.append(x)
        return result_states


# OU process
class sto_Net2(nn.Module):  
    def __init__(self, activation='heav', dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(sto_Net2, self).__init__()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.dim_out = dim_out
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        if activation == 'heav':
            self.act = lambda x: (x > 0).float()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise NameError('No such activations.')

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1])):
        assert Cs_list.shape[0] == len(self.linear_sequence)
        # the layer before mnn
        states = []
        x = img.view(-1, self.dim_in) + torch.randn_like(img.view(-1, self.dim_in)) * np.sqrt(C_in)
        states.append(x)
        # first layer
        if len(self.linear_sequence) > 1:
            x = self.act(self.linear_sequence[0](x))
            x = x + torch.randn_like(x) * np.sqrt(Cs_list[0])
            states.append(x)
        # middle layers
        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x = self.act(linear_func(x))
            x = x + torch.randn_like(x) * np.sqrt(Cs_list[i + 1])
            states.append(x)
        # final layer
        x = self.act(self.linear_sequence[-1](x))
        x = x + torch.randn_like(x) * np.sqrt(Cs_list[-1])
        states.append(x)
        x = self.final(x)
        states.append(x)
        return states

    def run(self, dt, img, states, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1])):
        assert Cs_list.shape[0] == len(self.linear_sequence)
        result_states = []
        x = states[0] + dt * (img.view(-1, self.dim_in) - states[0]) + np.sqrt(2 * dt * C_in) * torch.randn_like(states[0])
        result_states.append(x)
        # first layer
        if len(self.linear_sequence) > 1:
            x = states[1] + dt * (self.act(self.linear_sequence[0](x)) - states[1]) + \
                np.sqrt(2*dt * Cs_list[0]) * torch.randn_like(states[1])
            result_states.append(x)
        # middle layers
        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x = states[i + 2] + dt * (self.act(linear_func(x)) - states[i + 2]) + \
                np.sqrt(2*dt * Cs_list[i + 2]) * torch.randn_like(states[i + 2])
            result_states.append(x)
        # final layer
        x = states[-2] + dt * (self.act(self.linear_sequence[-1](x)) - states[-2]) + \
            np.sqrt(2*dt * Cs_list[-1]) * torch.randn_like(states[-2])
        result_states.append(x)
        x = states[-1] + dt * (self.final(x) - states[-1])
        result_states.append(x)
        return result_states
