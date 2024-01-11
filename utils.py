import numpy as np
import torch


def entropy_cal(output_C, eps=1e-40):
    singular = torch.linalg.svd(output_C)[1]
    if len(output_C.shape)>2:
        dim_out = (singular > eps).float().sum(dim=1)
        singular[singular <= eps] = 1
        log_det = torch.log(singular).sum(dim=1)
        entropy = (dim_out / 2 * (1 + np.log(2 * np.pi)) + 0.5 * log_det).cpu().numpy()
        return entropy
    else:
        dim_out = (singular > eps).float().sum()
        singular[singular <= eps] = 1
        log_det = torch.log(singular).sum()
        entropy = (dim_out / 2 * (1 + np.log(2 * np.pi)) + 0.5 * log_det).cpu().numpy()
        return [entropy]

def ll_cal(output, y, variance,rescale=None):
    assert variance > 10e-10
    if rescale is None:
        return -0.5 * (np.log(2 * np.pi * variance) + (output - y) ** 2 / variance)
    else:
        return -0.5 * (np.log(2 * np.pi * variance * rescale**2) + (output - y) ** 2 / variance)

def stats_report(array, ignore_nan):
    if ignore_nan:
        return np.nanmean(array), np.nanvar(array)
    else:
        return np.mean(array), np.var(array)


def right_wrong_stats_report(array, flag, ignore_nan=True):
    right_mean, right_var = stats_report(array[flag > 0.5], ignore_nan=ignore_nan)
    wrong_mean, wrong_var = stats_report(array[flag < 0.5], ignore_nan=ignore_nan)
    return right_mean, right_var, wrong_mean, wrong_var


