import math
from mindspore import nn
import mindspore as ms
from collections import Counter
import numpy as np


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate."""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_step_lr(lr, lr_epochs, steps_per_epoch, warmup_epochs, max_epoch, gamma=0.1):
    """Warmup step learning rate."""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    milestones = lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone * steps_per_epoch
        milestones_steps.append(milestones_step)

    lr_each_step = []
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma**milestones_steps_counter[i]
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr_sample(lr, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """Warmup cosine annealing learning rate."""
    start_sample_epoch = 60
    step_sample = 2
    tobe_sampled_epoch = 60
    end_sampled_epoch = start_sample_epoch + step_sample * tobe_sampled_epoch
    max_sampled_epoch = max_epoch + tobe_sampled_epoch
    T_max = max_sampled_epoch

    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    total_sampled_steps = int(max_sampled_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []

    for i in range(total_sampled_steps):
        last_epoch = i // steps_per_epoch
        if last_epoch in range(start_sample_epoch, end_sampled_epoch, step_sample):
            continue
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr)

    assert total_steps == len(lr_each_step)
    return np.array(lr_each_step).astype(np.float32)


def get_lr(cfg,steps_per_epoch):
    # enable warmup
    warmup_epochs = cfg.SOLVER.WARMUP_EPOCH
    if cfg.SOLVER.LR_WARMUP:
        # assert warmup_scheduler is not None
        lr = warmup_cosine_annealing_lr_sample(cfg.SOLVER.BASE_LR, steps_per_epoch, warmup_epochs,
                                               300, cfg.SOLVER.WARMUP_STEPS,
                                               cfg.SOLVER.BASE_LR / cfg.SOLVER.DIV_FACTOR)
    else:
        lr = warmup_step_lr(cfg.SOLVER.BASE_LR, cfg.SOLVER.DECAY_EPOCH_STEPS, steps_per_epoch, warmup_epochs, 300,
                            gamma=cfg.SOLVER.LR_DECAY)

    return lr

def get_optim(cfg,net,steps_per_epoch):
    optim_cfg = cfg.SOLVER
    params = get_param_groups(net,cfg)
    # lr=get_lr(cfg,steps_per_epoch)

    # if optim_cfg.OPTIMIZER != 'adam_onecycle':
    #     model_params = get_param_groups(net)

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = nn.Adam(params, learning_rate=optim_cfg.BASE_LR, weight_decay=optim_cfg.WEIGHT_DECAY,
                            beta1=0.9, beta2=0.99)

    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = nn.AdamWeightDecay(params, learning_rate=optim_cfg.BASE_LR,
                                       weight_decay=optim_cfg.WEIGHT_DECAY,
                                       beta1=0.9, beta2=0.99)

    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = nn.SGD(
            params, learning_rate=ms.Tensor(optim_cfg.BASE_LR,ms.float32), weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )

    else:
        raise NotImplementedError

    return optimizer


def get_param_groups(network,cfg):
    """ get param groups """
    params = []
    beta_params=[]
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith(".beta"):
            # all beta using gynamic lr
            beta_params.append(x)
        else:params.append(x)

    return [{'params': params, 'weight_decay': 0.0,'lr':cfg.SOLVER.BASE_LR}, {'params': beta_params, 'lr':cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR}]