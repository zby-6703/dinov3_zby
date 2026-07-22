from __future__ import annotations

"""Multi-stage training helpers for character / waterline / draft-depth tasks."""

from typing import Any, Dict, Optional


DEFAULT_TASK_TRAIN = {
    "stage": "joint",
    "train_character": True,
    "train_waterline": True,
    "train_depth": True,
    "train_structure": True,
    "freeze": {
        "backbone": False,
        "encoder": False,
        "decoder": False,
        "head": False,
        "direct_depth_head": False,
    },
}


def resolve_task_train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(cfg.get("TaskTrain") or {})
    freeze_raw = dict(DEFAULT_TASK_TRAIN["freeze"])
    freeze_raw.update(dict(raw.get("freeze") or {}))
    resolved = dict(DEFAULT_TASK_TRAIN)
    resolved.update({k: v for k, v in raw.items() if k != "freeze"})
    resolved["freeze"] = freeze_raw

    # Keep freeze flags consistent with task switches.
    if not resolved["train_depth"]:
        resolved["freeze"]["direct_depth_head"] = True
    if not resolved["train_character"] and not resolved["train_waterline"]:
        # Depth-only stage: freeze perception stack by default if user omitted freeze.
        user_freeze = dict(raw.get("freeze") or {})
        for key in ("backbone", "encoder", "decoder", "head"):
            if key not in user_freeze:
                resolved["freeze"][key] = True
    return resolved


def apply_task_train_policy(model, cfg: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Apply TaskTrain switches: freeze modules and store flags on model/loss."""
    policy = resolve_task_train(cfg)
    model.train_character = bool(policy["train_character"])
    model.train_waterline = bool(policy["train_waterline"])
    model.train_depth = bool(policy["train_depth"])
    model.train_structure = bool(policy.get("train_structure", True))

    if getattr(model, "loss", None) is not None:
        model.loss.train_character = model.train_character
        model.loss.train_waterline = model.train_waterline

    modules = {
        "backbone": getattr(model, "backbone", None),
        "encoder": getattr(model, "encoder", None),
        "decoder": getattr(model, "decoder", None),
        "head": getattr(model, "head", None),
        "direct_depth_head": getattr(model, "direct_depth_head", None),
    }
    freeze_cfg = policy["freeze"]
    frozen_modules = []
    for name, module in modules.items():
        if module is None:
            continue
        frozen = bool(freeze_cfg.get(name, False))
        # If depth is disabled entirely, force-freeze when module exists.
        if name == "direct_depth_head" and not model.train_depth:
            frozen = True
        for param in module.parameters():
            param.requires_grad = not frozen
        if frozen:
            module.eval()
            frozen_modules.append(name)
        if logger is not None:
            n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in module.parameters())
            logger.info(
                "TaskTrain freeze[%s]=%s trainable_params=%d/%d",
                name,
                frozen,
                n_train,
                n_total,
            )

    model._task_frozen_modules = tuple(frozen_modules)

    if logger is not None:
        logger.info(
            "TaskTrain stage=%s character=%s waterline=%s depth=%s structure=%s",
            policy.get("stage"),
            model.train_character,
            model.train_waterline,
            model.train_depth,
            model.train_structure,
        )
    return policy


def uses_partial_training(cfg: Dict[str, Any]) -> bool:
    """Whether DDP should enable find_unused_parameters."""
    policy = resolve_task_train(cfg)
    if not (policy["train_character"] and policy["train_waterline"] and policy["train_depth"]):
        return True
    freeze = policy["freeze"]
    return any(bool(v) for v in freeze.values())
