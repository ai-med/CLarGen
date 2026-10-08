"""Configuration handling: YAML defaults (``config/config.yaml``) + command-line overrides."""
import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_by_path(cfg, dotted_key, value):
    """Set ``cfg["a"]["b"] = value`` for ``dotted_key = "a.b"``."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"Unknown config section '{key}' in '{dotted_key}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[keys[-1]] = value


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Path to the YAML configuration file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override any config entry, e.g. --set retrieval.alpha=0.7 generation.top_p=0.9")
    return parser


def resolve_config(args, shortcuts):
    """Load the YAML config and apply overrides.

    ``shortcuts`` maps argparse destinations to dotted config keys; an argument
    left at ``None`` keeps the YAML value. ``--set`` overrides are applied last.
    """
    cfg = copy.deepcopy(load_config(args.config))
    for dest, dotted_key in shortcuts.items():
        value = getattr(args, dest, None)
        if value is not None:
            set_by_path(cfg, dotted_key, value)
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        set_by_path(cfg, key, yaml.safe_load(value))
    return cfg


def resolve_path(path):
    """Resolve a config path relative to the repository root."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
