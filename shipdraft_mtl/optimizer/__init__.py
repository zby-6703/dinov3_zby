import copy

import torch
from torch import nn

__all__ = ["build_optimizer"]


def _build_param_groups(model: nn.Module, lr: float, weight_decay: float, filter_bias_and_bn: bool, backbone_lr_mult: float):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_lr = lr * backbone_lr_mult if "backbone" in name else lr
        group_weight_decay = weight_decay
        if filter_bias_and_bn and (param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower()):
            group_weight_decay = 0.0
        params.append({"params": [param], "lr": group_lr, "weight_decay": group_weight_decay})
    return params


def build_optimizer(optim_config, lr_scheduler_config, epochs, step_each_epoch, model):
    from . import lr as lr_module

    config = copy.deepcopy(optim_config)
    optimizer_name = config.pop("name")
    base_lr = config.pop("lr")
    weight_decay = config.pop("weight_decay", 0.0)
    filter_bias_and_bn = config.pop("filter_bias_and_bn", False)
    backbone_lr_mult = config.pop("backbone_lr_mult", 1.0)

    if isinstance(model, nn.Module):
        parameters = _build_param_groups(
            model,
            lr=base_lr,
            weight_decay=weight_decay,
            filter_bias_and_bn=filter_bias_and_bn,
            backbone_lr_mult=backbone_lr_mult,
        )
    else:
        parameters = model

    optimizer = getattr(torch.optim, optimizer_name)(
        params=parameters,
        lr=base_lr,
        weight_decay=0.0 if isinstance(parameters, list) else weight_decay,
        **config,
    )

    lr_config = copy.deepcopy(lr_scheduler_config)
    scheduler_name = lr_config.pop("name")
    lr_config.update({"epochs": epochs, "step_each_epoch": step_each_epoch, "lr": base_lr})
    lr_scheduler = getattr(lr_module, scheduler_name)(**lr_config)(optimizer=optimizer)
    return optimizer, lr_scheduler
