# -*- coding:utf-8 -*-

import torch

def get_random_fluctuation(values, noise_std, device):
    """Return generation performance data with random noise."""
    noise = torch.normal(mean = 0, std = noise_std, size = values.shape).to(device)
    return values + noise
