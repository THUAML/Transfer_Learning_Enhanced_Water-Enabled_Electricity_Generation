# -*- coding: utf-8 -*-

def lr_scheduler(optimizer, iter_num, gamma=1e-4, power=0.75, wei_decay = 1.0, weight_decay=0.0, min_lr = 1e-6):
    lr_decay = ((1 + gamma * (iter_num-1))/(1 + gamma * iter_num)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay#max(param_group['lr'] * lr_decay, min_lr)
        param_group['weight_decay'] = weight_decay * wei_decay
    return optimizer