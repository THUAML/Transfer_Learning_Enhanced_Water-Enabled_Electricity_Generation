# -*- coding: utf-8 -*-

import torch

def loss_F(pred, target, device=torch.device(0), weight=None, coeff=0):
    assert pred.size() == target.size()
    if weight == None:
        weight = torch.ones(pred.size())
    weight = weight.to(device)
    loss = torch.mul(weight, torch.pow(pred - target, 2))
    return torch.mean(loss)