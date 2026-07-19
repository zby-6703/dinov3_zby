import copy
from importlib import import_module

__all__ = ["build_head"]

name_to_module = {
    "DraftFormerPredictionHead": ".draftformer_prediction_head",
}


def build_head(config, num_decoder_layers):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported head: {module_name}")
    config["num_decoder_layers"] = num_decoder_layers
    module = import_module(name_to_module[module_name], package=__package__)
    module_class = getattr(module, module_name)
    return module_class(**config)
