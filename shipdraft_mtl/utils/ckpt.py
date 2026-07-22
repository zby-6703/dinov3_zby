import os

import torch

from .logging import get_logger


def _torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        raise RuntimeError(
            "This PyTorch version cannot safely load checkpoints. Upgrade PyTorch "
            "to a version that supports torch.load(..., weights_only=True)."
        )


def _model_state_dict(model, distributed=False):
    return model.module.state_dict() if distributed and hasattr(model, "module") else model.state_dict()


def save_ckpt(
    model,
    cfg,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    metrics,
    is_best=False,
    logger=None,
    prefix=None,
):
    if logger is None:
        logger = get_logger()
    if prefix is None:
        filename = "best.pth" if is_best else "latest.pth"
    else:
        filename = prefix + ".pth"
    save_path = os.path.join(cfg["Global"]["output_dir"], filename)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": _model_state_dict(model, cfg["Global"].get("distributed", False)),
        "optimizer": None if is_best or optimizer is None else optimizer.state_dict(),
        "scheduler": None if is_best or lr_scheduler is None else lr_scheduler.state_dict(),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "metrics": metrics,
    }
    torch.save(state, save_path)
    logger.info(f"save ckpt to {save_path}")


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    if logger is None:
        logger = get_logger()
    checkpoints = cfg["Global"].get("checkpoints")
    pretrained_model = cfg["Global"].get("pretrained_model")

    status = {}
    if checkpoints and os.path.exists(checkpoints):
        checkpoint = _torch_load_checkpoint(checkpoints, map_location=torch.device("cpu"))
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and checkpoint.get("scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"resume from checkpoint {checkpoints} (epoch {checkpoint.get('epoch', 0)})")
        status["global_step"] = checkpoint.get("global_step", 0)
        status["epoch"] = checkpoint.get("epoch", 0) + 1
        status["metrics"] = checkpoint.get("metrics", {})
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model, logger)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def load_pretrained_params(model, pretrained_model, logger=None):
    if logger is None:
        logger = get_logger()
    # Avoid Windows escape corruption (e.g. "\best.pth" -> backspace + "est.pth").
    pretrained_model = os.path.normpath(str(pretrained_model).replace("\\", "/"))
    if not os.path.isfile(pretrained_model):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_model}")
    checkpoint = _torch_load_checkpoint(pretrained_model, map_location=torch.device("cpu"))
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    matched_state = {}
    skipped_mismatch = []
    skipped_unexpected = []

    for name, tensor in state_dict.items():
        clean_name = name[7:] if name.startswith("module.") else name
        if clean_name not in model_state:
            skipped_unexpected.append(clean_name)
            continue
        if model_state[clean_name].shape != tensor.shape:
            skipped_mismatch.append((clean_name, tuple(tensor.shape), tuple(model_state[clean_name].shape)))
            continue
        matched_state[clean_name] = tensor

    missing_keys = [name for name in model_state.keys() if name not in matched_state]
    model.load_state_dict(matched_state, strict=False)
    logger.info(f"Loaded {len(matched_state)}/{len(model_state)} params from pretrained checkpoint")
    if skipped_mismatch:
        logger.info(f"Skipped {len(skipped_mismatch)} params due to shape mismatch")
    if skipped_unexpected:
        logger.info(f"Skipped {len(skipped_unexpected)} unexpected params not found in model")
    if missing_keys:
        logger.info(f"Model has {len(missing_keys)} params not initialized from checkpoint")
