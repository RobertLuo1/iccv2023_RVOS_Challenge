import os
from os import path
import datetime
import shutil
import torch
import numpy as np
import math


def flatten_temporal_batch_dims(outputs, targets):
    for k in outputs.keys():
        if k == 'pred_logit' or k == 'text_sentence_feature' or k == 'pred_is_referred':
            continue
        if isinstance(outputs[k], torch.Tensor):
            outputs[k] = outputs[k].flatten(0, 1)
        else:  # list
            outputs[k] = [i for step_t in outputs[k] for i in step_t]
    targets = [frame_t_target for step_t in targets for frame_t_target in step_t] #[({}, {})]
    return outputs, targets


def create_output_dir(config):
    root = '/mnt/data_16TB/lzy23'
    output_dir_path = path.join(root, 'runs', config.dataset_name, config.version)
    os.makedirs(output_dir_path, exist_ok=True)
    shutil.copyfile(src=config.config_path, dst=path.join(output_dir_path, 'config.yaml'))
    return output_dir_path


def create_checkpoint_dir(output_dir_path):
    checkpoint_dir_path = path.join(output_dir_path, 'checkpoints')
    os.makedirs(checkpoint_dir_path, exist_ok=True)
    return checkpoint_dir_path

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda