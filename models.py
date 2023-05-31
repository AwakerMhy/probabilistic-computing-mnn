import numpy as np
import torch
import torch.nn as nn
from activations import Heaviside, Relu


# batch-wise covariance
class Net(nn.Module):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(Net, self).__init__()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.dim_out = dim_out
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9

    def cov_forward(self, ubar, Vbar, Cbar, no_cov=False):
        V_activated = self.C_v(ubar, Vbar)
        if no_cov:
            cov_activated = torch.zeros_like(Cbar)
            cov_activated[
                torch.arange(cov_activated.shape[0]), torch.arange(cov_activated.shape[0])] = V_activated
        else:
            p = torch.zeros_like(V_activated).reshape(1, -1)
            p[Vbar.reshape(1, -1) > self.eps] = self.psi(ubar.reshape(1, -1)[Vbar.reshape(1, -1) > self.eps],
                                                         Vbar.reshape(1, -1)[Vbar.reshape(1, -1) > self.eps]
                                                         ) / torch.sqrt(
                Vbar.reshape(1, -1)[Vbar.reshape(1, -1) > self.eps])
            cov_activated = p * p.reshape(-1, 1) * Cbar
            cov_activated[
                torch.arange(cov_activated.shape[0]), torch.arange(cov_activated.shape[0])] = V_activated

            return cov_activated

    def forward_layer(self, x, linear_func, Cov_in, Cs, cal_cov=False, no_cov=False):
        with torch.no_grad():
            ubar = linear_func(x.mean(dim=0))
            w = linear_func.weight.data
            if no_cov:
                Cbar = (w * w * Cov_in).sum(dim=1)
                Vbar = Cbar + Cs
                x = self.m_v(linear_func(x), Vbar.unsqueeze(0))
                if cal_cov:
                    return x, Vbar
                else:
                    return x
            else:
                if len(Cov_in.shape) == 0:
                    Cbar = torch.matmul(w, w.T) * Cov_in
                else:
                    Cbar = torch.matmul(torch.matmul(w, Cov_in), w.T)
                Vbar = torch.diag(Cbar) + Cs
            if cal_cov:
                cov_activated = self.cov_forward(ubar, Vbar, Cbar, no_cov=no_cov)
        x = self.m_v(linear_func(x), Vbar.unsqueeze(0))
        if cal_cov:
            return x, cov_activated
        else:
            return x

    def forward_before(self, img):
        return img.view(-1, self.dim_in)

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1]),
                output_cov_activated_mid=False, output_C=False, no_cov=False):

        assert Cs_list.shape[0] == len(self.linear_sequence)

        x = self.forward_before(img)

        if output_cov_activated_mid:
            output_cov_activated = []

        # first layer
        if len(self.linear_sequence) > 1:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[0], C_in, Cs_list[0], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
        else:
            cov_activated = C_in

        # middle layers
        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x, cov_activated = self.forward_layer(x, linear_func, cov_activated, Cs_list[i + 1], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)

        # final layer
        if output_C:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1],
                                                  cal_cov=True, no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
            x = self.final(x)
            with torch.no_grad():
                w = self.final.weight.data
                if no_cov:
                    C_final = (w * w * cov_activated).sum(dim=1)
                else:
                    C_final = torch.matmul(torch.matmul(w, cov_activated), w.T)
        else:
            x = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1], cal_cov=False,
                                   no_cov=no_cov)
            x = self.final(x)

        if output_cov_activated_mid:
            if output_C:
                return x, output_cov_activated, C_final
            else:
                return x, output_cov_activated
        else:
            if output_C:
                return x, C_final
            else:
                return x


# no batch-wise covariance
class Net2(nn.Module):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(Net2, self).__init__()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.dim_out = dim_out
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9

    def cov_forward(self, ubar, Vbar, Cbar, no_cov=False):
        V_activated = self.C_v(ubar, Vbar)
        if no_cov:
            cov_activated = torch.zeros_like(Cbar) + torch.diag_embed(V_activated, dim1=-2, dim2=-1)
        else:
            p = torch.zeros_like(V_activated)
            p[Vbar > self.eps] = self.psi(ubar[Vbar > self.eps], Vbar[Vbar > self.eps]) / torch.sqrt(
                Vbar[Vbar > self.eps])

            cov_activated = torch.bmm(p.reshape(p.shape[0], p.shape[1], 1), p.reshape(p.shape[0], 1, p.shape[1])) * Cbar

            mask = torch.eye(cov_activated.shape[1], device=cov_activated.device).bool().unsqueeze(0).expand(
                cov_activated.shape[0], -1, -1)
            cov_activated.masked_fill_(mask, 0)
            cov_activated = cov_activated + torch.diag_embed(V_activated, dim1=-2, dim2=-1)
        return cov_activated

    def forward_layer(self, x, linear_func, Cov_in, Cs, cal_cov=False, no_cov=False):
        with torch.no_grad():
            ubar = linear_func(x)
            w = linear_func.weight.data
            if len(Cov_in.shape) == 0:
                Cbar = (torch.matmul(w, w.T) * Cov_in).expand(ubar.shape[0], -1, -1)
            else:
                Cbar = torch.bmm(torch.bmm(w.expand(Cov_in.shape[0], -1, -1), Cov_in),
                                 w.T.expand(Cov_in.shape[0], -1, -1))
            Vbar = torch.diagonal(Cbar, dim1=-2, dim2=-1) + Cs
            if cal_cov:
                cov_activated = self.cov_forward(ubar, Vbar, Cbar, no_cov=no_cov)
        x = self.m_v(linear_func(x), Vbar)
        if cal_cov:
            return x, cov_activated
        else:
            return x

    def forward_before(self, img):
        return img.view(-1, self.dim_in)

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1]),
                output_cov_activated_mid=False, output_C=False, no_cov=False):

        assert Cs_list.shape[0] == len(self.linear_sequence)

        x = self.forward_before(img)

        if output_cov_activated_mid:
            output_cov_activated = []

        if len(self.linear_sequence) > 1:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[0], C_in, Cs_list[0], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
        else:
            cov_activated = C_in

        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x, cov_activated = self.forward_layer(x, linear_func, cov_activated, Cs_list[i + 1], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)

        if output_C:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1],
                                                  cal_cov=True, no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
            x = self.final(x)
            with torch.no_grad():
                w = self.final.weight.data
                C_final = torch.matmul(torch.matmul(w, cov_activated), w.T)
        else:
            x = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1], cal_cov=False,
                                   no_cov=no_cov)
            x = self.final(x)

        if output_cov_activated_mid:
            if output_C:
                return x, output_cov_activated, C_final
            else:
                return x, output_cov_activated
        else:
            if output_C:
                return x, C_final
            else:
                return x


class HeavisideNet(Heaviside, Net):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(HeavisideNet, self).__init__()
        self.dim_in = dim_s[0]
        self.linear_sequence = nn.ModuleList()
        self.layer_num = dim_s.shape[0]
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9


class HeavisideNet2(Heaviside, Net2):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(HeavisideNet2, self).__init__()
        self.dim_in = dim_s[0]
        self.linear_sequence = nn.ModuleList()
        self.layer_num = dim_s.shape[0]
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9


class ReluNet(Relu, Net):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(ReluNet, self).__init__()
        self.dim_in = dim_s[0]
        self.linear_sequence = nn.ModuleList()
        self.layer_num = dim_s.shape[0]
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9


class ReluNet2(Relu, Net2):
    def __init__(self, dim_s=np.array([28 * 28, 392, 196, 96, 48, 24]), dim_out=10):
        super(ReluNet2, self).__init__()
        self.dim_in = dim_s[0]
        self.linear_sequence = nn.ModuleList()
        self.layer_num = dim_s.shape[0]
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9


class CNN_Net(Net):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s
        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias

    def forward_before(self, img, bn_s=None, pool_s=None):
        x = img
        if bn_s is None:
            bn_s = [False] * self.conv_num
        if pool_s is None:
            pool_s = [True] * self.conv_num

        bn_index = 0
        pool_index = 0
        for i in range(self.conv_num):
            x = self.conv_sequence[i](x)
            if bn_s[i]:
                x = self.bn2d[bn_index](x)
                bn_index += 1
            x = self.relu(x)
            if pool_s[i]:
                x = self.pools[pool_index](x)
                pool_index += 1
        return x.view(x.size(0), -1)

    def forward_layer_cnn(self, x, Cov_in, conv2, Cs=torch.tensor(0), cal_cov=False, no_cov=False, pool=None, bn=None):
        with torch.no_grad():
            ubar = conv2(x.mean(dim=0, keepdims=True)).squeeze(0)
            if self.use_bias:
                bias = conv2.bias.data
                conv2.bias.data *= 0

            dim_in = x.shape[1] * x.shape[2] * x.shape[3]
            dim_out = ubar.shape[0] * ubar.shape[1] * ubar.shape[2]

            if no_cov:
                raise NotImplementedError('TODO')
            else:
                if len(Cov_in.shape) == 0:
                    Cov_in = Cov_in * torch.eye(dim_in).to(ubar.device)
                    Cov_in = Cov_in.reshape(x.shape[1], x.shape[2], x.shape[3], x.shape[1], x.shape[2], x.shape[3])

                Cbar_ = conv2(Cov_in.permute(3, 4, 5, 0, 1, 2).reshape(dim_in, x.shape[1], x.shape[2], x.shape[3]))
                Cbar = conv2(Cbar_.permute(1, 2, 3, 0).reshape(dim_out, x.shape[1], x.shape[2], x.shape[3]))
                Vbar = torch.diag(Cbar.reshape(dim_out, dim_out)).reshape(ubar.shape) + Cs
                if cal_cov:
                    cov_activated = self.cov_forward(ubar.reshape(dim_out), Vbar.reshape(dim_out),
                                                     Cbar.reshape(dim_out, dim_out)).reshape(
                        ubar.shape[0], ubar.shape[1], ubar.shape[2], ubar.shape[0], ubar.shape[1], ubar.shape[2])
                if self.use_bias:
                    conv2.bias.data = bias
        x = self.m_v(conv2(x), Vbar.unsqueeze(0))

        if bn is not None:
            x = bn(x)
            if cal_cov:
                with torch.no_grad():
                    std = bn.weight / torch.sqrt(bn.running_var + self.eps)
                    cov_activated *= std.reshape(-1, 1, 1, 1, 1, 1)
                    cov_activated *= std.reshape(1, 1, 1, -1, 1, 1)

        if pool is not None:
            x = pool(x)
            with torch.no_grad():
                cov_activated_pool = pool(
                    cov_activated.reshape(dim_out, ubar.shape[0], ubar.shape[1], ubar.shape[2]))
                dim_out_pool = np.prod(cov_activated_pool.shape[1:])
                cov_activated = pool(
                    cov_activated_pool.permute(1, 2, 3, 0).reshape(dim_out_pool, ubar.shape[0], ubar.shape[1],
                                                                   ubar.shape[2]))
                cov_activated = cov_activated.reshape(x.shape[1], x.shape[2], x.shape[3],
                                                      x.shape[1], x.shape[2], x.shape[3])

        if cal_cov:
            return x, cov_activated
        else:
            return x

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1]), Cs_conv=np.array([1, 1]),
                output_cov_activated_mid=False, output_C=False, no_cov=False):
        assert Cs_list.shape[0] == len(self.linear_sequence)

        output_cov_activated = []
        x = self.forward_before(img, bn_s=self.bn_s, pool_s=self.pool_s)
        cov_activated = C_in

        x = x.reshape(x.size(0), -1)
        dim_x = x.shape[1]
        if len(cov_activated.shape) > 1:
            cov_activated = cov_activated.reshape(dim_x, dim_x)

        if self.layer_num > 2:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[0], cov_activated, Cs_list[0], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)

        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x, cov_activated = self.forward_layer(x, linear_func, cov_activated, Cs_list[i + 1], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
        if output_C:
            if self.layer_num > 1:
                x, cov_activated = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1],
                                                      cal_cov=True, no_cov=no_cov)
                if output_cov_activated_mid:
                    output_cov_activated.append(cov_activated)
            x = self.final(x)
            with torch.no_grad():
                w = self.final.weight.data
                if len(cov_activated.shape) == 0:
                    C_final = torch.matmul(w, w.T) * cov_activated
                else:
                    C_final = torch.matmul(torch.matmul(w, cov_activated), w.T)
        else:
            if self.layer_num > 1:
                x = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1], cal_cov=False,
                                       no_cov=no_cov)
            x = self.final(x)

        if output_cov_activated_mid:
            if output_C:
                return x, output_cov_activated, C_final
            else:
                return x, output_cov_activated
        else:
            if output_C:
                return x, C_final
            else:
                return x


class CNN_HeavisideNet(HeavisideNet, CNN_Net):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s
        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias


class CNN_ReluNet(ReluNet, CNN_Net):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s

        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias


# no batch-wise covariance
class CNN_Net2(Net2):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net2, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s

        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias

    def forward_before(self, img, bn_s=None, pool_s=None):
        x = img
        if bn_s is None:
            bn_s = [False] * self.conv_num
        if pool_s is None:
            pool_s = [True] * self.conv_num

        bn_index = 0
        pool_index = 0
        for i in range(self.conv_num):
            x = self.conv_sequence[i](x)
            if bn_s[i]:
                x = self.bn2d[bn_index](x)
                bn_index += 1
            x = self.relu(x)
            if pool_s[i]:
                x = self.pools[pool_index](x)
                pool_index += 1
        return x.view(x.size(0), -1)

    def forward_layer_cnn(self, x, Cov_in, conv2, Cs=torch.tensor(0), cal_cov=False, no_cov=False, pool=None, bn=None):
        with torch.no_grad():
            ubar = conv2(x.mean(dim=0, keepdims=True)).squeeze(0)
            if self.use_bias:
                bias = conv2.bias.data
                conv2.bias.data *= 0

            dim_in = x.shape[1] * x.shape[2] * x.shape[3]
            dim_out = ubar.shape[0] * ubar.shape[1] * ubar.shape[2]

            if no_cov:
                raise NotImplementedError('TODO')
            else:
                if len(Cov_in.shape) == 0:
                    Cov_in = Cov_in * torch.eye(dim_in).to(ubar.device)
                    Cov_in = Cov_in.reshape(x.shape[1], x.shape[2], x.shape[3], x.shape[1], x.shape[2], x.shape[3])
                Cbar_ = conv2(Cov_in.permute(3, 4, 5, 0, 1, 2).reshape(dim_in, x.shape[1], x.shape[2], x.shape[3]))
                Cbar = conv2(Cbar_.permute(1, 2, 3, 0).reshape(dim_out, x.shape[1], x.shape[2], x.shape[3]))
                Vbar = torch.diag(Cbar.reshape(dim_out, dim_out)).reshape(ubar.shape) + Cs
                if cal_cov:
                    cov_activated = self.cov_forward(ubar.reshape(dim_out), Vbar.reshape(dim_out),
                                                     Cbar.reshape(dim_out, dim_out)).reshape(
                        ubar.shape[0], ubar.shape[1], ubar.shape[2], ubar.shape[0], ubar.shape[1], ubar.shape[2])
                if self.use_bias:
                    conv2.bias.data = bias
        x = self.m_v(conv2(x), Vbar.unsqueeze(0))

        if bn is not None:
            x = bn(x)
            if cal_cov:
                with torch.no_grad():
                    std = bn.weight / torch.sqrt(bn.running_var + self.eps)
                    cov_activated *= std.reshape(-1, 1, 1, 1, 1, 1)
                    cov_activated *= std.reshape(1, 1, 1, -1, 1, 1)
        if pool is not None:
            x = pool(x)
            with torch.no_grad():
                cov_activated_pool = pool(
                    cov_activated.reshape(dim_out, ubar.shape[0], ubar.shape[1], ubar.shape[2]))
                dim_out_pool = np.prod(cov_activated_pool.shape[1:])
                cov_activated = pool(
                    cov_activated_pool.permute(1, 2, 3, 0).reshape(dim_out_pool, ubar.shape[0], ubar.shape[1],
                                                                   ubar.shape[2]))
                cov_activated = cov_activated.reshape(x.shape[1], x.shape[2], x.shape[3],
                                                      x.shape[1], x.shape[2], x.shape[3])

        if cal_cov:
            return x, cov_activated
        else:
            return x

    def forward(self, img, C_in=torch.tensor(1), Cs_list=np.array([1, 1, 1, 1, 1]), Cs_conv=np.array([1, 1]),
                output_cov_activated_mid=False, output_C=False, no_cov=False):

        assert Cs_list.shape[0] == len(self.linear_sequence)
        output_cov_activated = []
        x = self.forward_before(img, bn_s=self.bn_s, pool_s=self.pool_s)
        cov_activated = C_in
        x = x.reshape(x.size(0), -1)
        dim_x = x.shape[1]
        if len(cov_activated.shape) > 1:
            cov_activated = cov_activated.reshape(dim_x, dim_x)
        if self.layer_num > 2:
            x, cov_activated = self.forward_layer(x, self.linear_sequence[0], cov_activated, Cs_list[0], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
        for i, linear_func in enumerate(self.linear_sequence[1:-1]):
            x, cov_activated = self.forward_layer(x, linear_func, cov_activated, Cs_list[i + 1], cal_cov=True,
                                                  no_cov=no_cov)
            if output_cov_activated_mid:
                output_cov_activated.append(cov_activated)
        if output_C:
            if self.layer_num > 1:
                x, cov_activated = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1],
                                                      cal_cov=True, no_cov=no_cov)
                if output_cov_activated_mid:
                    output_cov_activated.append(cov_activated)
            x = self.final(x)
            with torch.no_grad():
                w = self.final.weight.data
                if len(cov_activated.shape) == 0:
                    C_final = torch.matmul(w, w.T) * cov_activated
                else:
                    C_final = torch.matmul(torch.matmul(w, cov_activated), w.T)
        else:
            if self.layer_num > 1:
                x = self.forward_layer(x, self.linear_sequence[-1], cov_activated, Cs_list[-1], cal_cov=False,
                                       no_cov=no_cov)
            x = self.final(x)

        if output_cov_activated_mid:
            if output_C:
                return x, output_cov_activated, C_final
            else:
                return x, output_cov_activated
        else:
            if output_C:
                return x, C_final
            else:
                return x

# no batch-wise covariance
class CNN_HeavisideNet2(HeavisideNet2, CNN_Net2):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net2, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s
        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias


# no batch-wise covariance
class CNN_ReluNet2(ReluNet2, CNN_Net2):
    def __init__(self, dim_s, dim_out=10, conv_settings=((3, 6, 3, 1, 1), (6, 12, 3, 1, 1)), use_bias=False,
                 pooltype='max', bn_s=None, pool_s=None):
        super(CNN_Net2, self).__init__()
        self.bn_s = bn_s
        self.pool_s = pool_s
        self.bn2d = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.bn_s)):
                if bn_s[i]:
                    self.bn2d.append(nn.BatchNorm2d(conv_settings[i][1]))

        self.pools = nn.ModuleList()
        if self.bn_s is not None:
            for i in range(len(self.pool_s)):
                if pool_s[i]:
                    if pooltype == 'max':
                        self.pools.append(nn.MaxPool2d(kernel_size=2))
                    elif pooltype == 'ave':
                        self.pools.append(nn.AvgPool2d(kernel_size=2))

        self.conv_num = len(conv_settings)
        self.conv_sequence = nn.ModuleList()

        for i, conv_setting in enumerate(conv_settings):
            self.conv_sequence.append(torch.nn.Conv2d(
                in_channels=conv_setting[0], out_channels=conv_setting[1], kernel_size=conv_setting[2],
                stride=conv_setting[3],
                padding=conv_setting[4], bias=use_bias))

        self.relu = nn.ReLU()
        self.dim_in = dim_s[0]
        self.layer_num = dim_s.shape[0]
        self.linear_sequence = nn.ModuleList()
        for i in range(self.layer_num - 1):
            self.linear_sequence.append(nn.Linear(dim_s[i], dim_s[i + 1]))
        self.final = nn.Linear(dim_s[-1], dim_out)
        self.eps = 10e-9
        self.use_bias = use_bias
