"""Config merge and validation (intentionally independent of PyTorch)."""

from copy import deepcopy
from pathlib import Path
import json

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"


def deep_merge(a: dict, b: dict) -> dict:
    result = deepcopy(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return result


def configure(
    experiment: str,
    *,
    config_path: str | None = None,
    data_root: str | None = None,
    generated_root: str | None = None,
    layout: str | None = None,
    model: str | None = None,
    seed: int | None = None,
    epochs: int | None = None,
    pretrained: bool | None = None,
    save_root: str | None = None,
) -> dict:
    conf = read_yaml(DEFAULT_CONFIG)
    if config_path:
        conf = deep_merge(conf, read_yaml(config_path))
    experiments = read_yaml(EXPERIMENT_CONFIG)
    if experiment not in experiments:
        raise ValueError(f"Unknown experiment {experiment}; choose {list(experiments)}")
    conf = deep_merge(conf, experiments[experiment])
    overrides = [
        ("data", "root", data_root),
        ("data", "synthetic_root", generated_root),
        ("data", "layout", layout),
        ("model", "name", model),
        ("model", "pretrained", pretrained),
        ("train", "seed", seed),
        ("train", "epochs", epochs),
        ("train", "save_root", save_root),
    ]
    for group, key, value in overrides:
        if value is not None:
            conf[group][key] = value
    conf["experiment"] = experiment
    validate_config(conf)
    return conf


def validate_config(conf: dict) -> None:
    data, train, syn = conf["data"], conf["train"], conf["synthetic"]
    if data["layout"] not in ("manifest", "split"):
        raise ValueError("data.layout must be manifest or split")
    if data["synthetic_format"] not in ("logits", "probabilities"):
        raise ValueError("data.synthetic_format must be explicit logits/probabilities")
    if syn["mode"] not in ("none", "hard", "soft"):
        raise ValueError("synthetic.mode must be none/hard/soft")
    if syn["weighting"] not in ("fixed", "naive", "meta"):
        raise ValueError("synthetic.weighting must be fixed/naive/meta")
    if syn["soft_loss"] not in ("kd", "softbce", "softbce_dice"):
        raise ValueError("synthetic.soft_loss must be kd/softbce/softbce_dice")
    # With mode=none, weighting is inert (useful when a local pilot config
    # sets meta globally but experiments B0/B1/B2 override only mode).
    if not (0 < syn["weight_init"] < syn["weight_max"]):
        raise ValueError("Require 0 < weight_init < weight_max")
    if not (1 <= int(syn["meta_every"])):
        raise ValueError("meta_every must be at least 1")
    if syn["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if train["batch_size_real"] <= 0 or train["batch_size_syn"] <= 0:
        raise ValueError("Batch sizes must be positive")
    if conf["model"]["name"] not in ("unet", "sovasegnet", "segformer_b2"):
        raise ValueError("Invalid model.name")
    if conf.get("augment") not in ("none", "flip", "combined"):
        raise ValueError("Invalid augmentation setting")
    if conf["experiment"].startswith("S") and conf["augment"] != "none":
        raise ValueError("Primary S experiments must NOT mix conventional augmentation")


def resolve_manifest(conf: dict) -> Path:
    path = Path(conf["data"]["manifest"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_config(conf: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding="utf-8")
