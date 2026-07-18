import copy
from importlib import import_module

__all__ = ["build_loss"]


name_to_module = {
    "DraftFormerLoss": ".draftformer_loss",
}


def build_loss(config, architecture_config):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported loss: {module_name}")
    module = import_module(name_to_module[module_name], package=__package__)
    module_builder = getattr(module, "build_loss")
    return module_builder(config, architecture_config)
