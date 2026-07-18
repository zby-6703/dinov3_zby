import copy
from importlib import import_module

__all__ = ["build_encoder"]


name_to_module = {
    "HybridEncoder": ".hybrid_encoder",
    "DraftFormerEncoder": ".draftformer_encoder",
}


def build_encoder(config, input_shape):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported encoder: {module_name}")
    config["input_shape"] = input_shape
    module = import_module(name_to_module[module_name], package=__package__)
    module_class = getattr(module, module_name)
    return module_class(**config)
