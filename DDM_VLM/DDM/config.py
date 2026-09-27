from pathlib import Path
import argparse

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent


def load_model_config(model_name, config_path=None):
    path = Path(config_path) if config_path else PACKAGE_ROOT / model_name / "config.yaml"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_get(config, dotted_path, default=""):
    value = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return default
        value = value[key]
    return value


def load_config_from_args(model_name):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the model config YAML file.",
    )
    known_args, _ = parser.parse_known_args()
    return parser, load_model_config(model_name, known_args.config)
