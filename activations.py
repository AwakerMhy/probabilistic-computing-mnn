import numpy as np
import torch
import torch.nn as nn


class Heaviside(nn.Module):
    def __init__(self):
        super(Heaviside, self).__init__()
        self.eps = 10e-50
    def m_v(self, mu, C=torch.tensor(1)):
        result = (1 + torch.erf(mu / torch.sqrt(2 * C + self.eps))) / 2
        result[(mu >= 0)*(C == 0)] = 1.
        result[(mu < 0) * (C == 0)] = 0.
        return result

    def C_v(self, mu, C=torch.tensor(1)):
        mu_activated = self.m_v(mu, C)
        return mu_activated - mu_activated ** 2

    def psi(self, mu, C=torch.tensor(1)):
        return 1 / np.sqrt(2 * np.pi) * torch.exp(-mu ** 2 / (C + self.eps) / 2)


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.eps = 10e-49

    def m_v(self, mu, C=torch.tensor(1)):
        C[C < self.eps] = self.eps
        return torch.sqrt(C / 2 / np.pi) * torch.exp(-mu ** 2 / 2 / C) + mu * (
                1 + torch.erf(mu / torch.sqrt(2 * C))) / 2

    def C_v(self, mu, C=torch.tensor(1)):
        C[C < self.eps] = self.eps
        result = (mu ** 2 + C) / 2 * (torch.erf(mu / torch.sqrt(C * 2)) + 1) + torch.sqrt(
            C / 2 / np.pi) * mu * torch.exp(
            -mu ** 2 / 2 / C) - self.m_v(mu, C) ** 2
        result[C < self.eps] = 0
        return result

    def psi(self, mu, C=torch.tensor(1)):
        result = (1 + torch.erf(mu / torch.sqrt(2 * C))) * torch.sqrt(C) / 2
        result[C < self.eps] = 0
        return result
