'''
Author: supermantx
Date: 2024-09-09 22:00:37
LastEditTime: 2024-09-12 16:10:33
Description: 
'''
import math


def adjust_learning_rate(optimizer, epoch, config):
    """
    Adjust learning rate
    """
    if epoch < config.warmup_epochs:
        lr = float(config.min_lr) + (config.max_lr - float(config.min_lr)) * epoch / config.warmup_epochs
    else:
        lr = float(config.min_lr) + (config.max_lr - float(config.min_lr)) * (1 + math.cos(
            (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs) * math.pi)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
