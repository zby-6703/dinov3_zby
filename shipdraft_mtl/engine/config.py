"""DraftFormer configuration system.

This mirrors the ShipNameRecognition/OpenOCR style:
- YAML configs with optional ``_BASE_`` inheritance.
- CLI ``-o A.B=value`` overrides.
- Recursive dicts with attribute-style access for model modules.
"""

from __future__ import annotations

import os
import ntpath
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Mapping

import yaml

__all__ = [
    "ArgsParser",
    "Config",
    "ConfigDict",
    "parse_override_options",
    "print_dict",
]


def parse_override_options(opts):
    config = {}
    if not opts:
        return config
    for item in opts:
        item = item.strip()
        key, value = item.split("=", 1)
        value = yaml.safe_load(value)
        cur = config
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return config


class ArgsParser(ArgumentParser):
    def __init__(self):
        super().__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", "--config-file", help="configuration file to use")
        self.add_argument("-o", "--opt", nargs="*", help="set configuration options")
        self.add_argument("--local_rank")
        self.add_argument("--local-rank")

    def parse_args(self, argv=None):
        args = super().parse_args(argv)
        if args.config is None:
            self.error("Please specify --config=configure_file_path.")
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        return parse_override_options(opts)


class ConfigDict(dict):
    def __init__(self, value=None, **kwargs):
        super().__init__()
        value = {} if value is None else value
        self.update(value, **kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def update(self, value=None, **kwargs):
        if value is None:
            value = {}
        for key, item in dict(value, **kwargs).items():
            self[key] = item

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, ConfigDict):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def to_dict(self):
        out = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [
                    item.to_dict() if isinstance(item, ConfigDict) else item
                    for item in value
                ]
            else:
                out[key] = value
        return out


def _merge_dict(config, merge_dct):
    for key, value in merge_dct.items():
        sub_keys = key.split(".")
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {".".join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], Mapping) and isinstance(value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def print_dict(cfg, print_func=print, delimiter=0):
    for key, value in sorted(cfg.items()):
        if isinstance(value, Mapping):
            print_func(f"{delimiter * ' '}{key} : ")
            print_dict(value, print_func, delimiter + 4)
        elif isinstance(value, list) and value and isinstance(value[0], Mapping):
            print_func(f"{delimiter * ' '}{key} : ")
            for item in value:
                print_dict(item, print_func, delimiter + 4)
        else:
            print_func(f"{delimiter * ' '}{key} : {value}")


class Config:
    def __init__(self, config_path, BASE_KEY="_BASE_"):
        self.BASE_KEY = BASE_KEY
        self.cfg = ConfigDict(self._load_config_with_base(config_path, loading=set()))

    def _load_config_with_base(self, file_path, loading):
        file_path = os.path.abspath(os.path.expanduser(os.fspath(file_path)))
        canonical_path = os.path.normcase(file_path)
        if canonical_path in loading:
            raise ValueError(f"Cyclic _BASE_ configuration detected at: {file_path}")
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [".yml", ".yaml"]:
            raise ValueError(f"Only YAML configuration files are supported: {file_path}")

        loading.add(canonical_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            if not isinstance(file_cfg, Mapping):
                raise ValueError(f"Configuration root must be a mapping: {file_path}")

            if self.BASE_KEY in file_cfg:
                all_base_cfg = {}
                base_value = file_cfg[self.BASE_KEY]
                base_ymls = [base_value] if isinstance(base_value, str) else list(base_value)
                for base_yml in base_ymls:
                    base_yml = os.path.expanduser(os.fspath(base_yml))
                    # ntpath handles drive-letter and UNC paths even when parsing on POSIX.
                    if not (os.path.isabs(base_yml) or ntpath.isabs(base_yml)):
                        base_yml = os.path.join(os.path.dirname(file_path), base_yml)
                    base_cfg = self._load_config_with_base(base_yml, loading)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)
                del file_cfg[self.BASE_KEY]
                file_cfg = _merge_dict(all_base_cfg, file_cfg)

            file_cfg["filename"] = os.path.splitext(os.path.split(file_path)[-1])[0]
            return file_cfg
        finally:
            loading.remove(canonical_path)

    def merge_dict(self, args):
        self.cfg = ConfigDict(_merge_dict(self.cfg, args))

    def print_cfg(self, print_func=print):
        print_func("----------- Config -----------")
        print_dict(self.cfg, print_func)
        print_func("---------------------------------------------")

    def save(self, path, cfg=None):
        cfg = self.cfg if cfg is None else ConfigDict(cfg)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
