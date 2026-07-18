import copy
from importlib import import_module

__all__ = ["build_decoder"]


name_to_module = {
    "DraftFormerDecoder": ".draftformer_decoder",
}


def build_decoder(config, in_channels, num_feature_levels):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported decoder: {module_name}")
    config["in_channels"] = in_channels
    config.setdefault("total_num_feature_levels", num_feature_levels)
    module = import_module(name_to_module[module_name], package=__package__)
    module_class = getattr(module, module_name)
    return module_class(**config)
