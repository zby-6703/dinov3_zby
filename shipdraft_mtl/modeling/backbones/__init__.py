import copy
from importlib import import_module

__all__ = ["build_backbone"]


name_to_module = {
    "ConvNeXtBackbone": ".dinov3_backbone",
    "DINOv3ConvNeXtBackbone": ".dinov3_backbone",
    "DINOv3ViTBackbone": ".dinov3_backbone",
    "ResNetBackbone": ".resnet_backbone",
    "ViTBackbone": ".dinov3_backbone",
}


def build_backbone(config, in_channels=3):
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    if module_name not in name_to_module:
        raise ValueError(f"Unsupported backbone: {module_name}")
    config["in_channels"] = in_channels
    module = import_module(name_to_module[module_name], package=__package__)
    module_class = getattr(module, module_name)
    return module_class(**config)
