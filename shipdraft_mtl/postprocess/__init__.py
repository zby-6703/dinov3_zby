import copy
from importlib import import_module

__all__ = ["build_post_process"]


name_to_module = {
    "DraftFormerPostProcess": ".draftformer_postprocess",
}


def build_post_process(config, architecture_config):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported post process: {module_name}")
    module = import_module(name_to_module[module_name], package=__package__)
    module_class = getattr(module, module_name)
    head_cfg = architecture_config["Head"]
    decoder_cfg = architecture_config["Decoder"]
    config.setdefault("num_detection_queries", decoder_cfg["num_detection_queries"])
    config.setdefault("num_det_classes", head_cfg["num_classes"])
    config.setdefault("num_seg_classes", head_cfg.get("num_classes_seg", 1))
    return module_class(**config)
