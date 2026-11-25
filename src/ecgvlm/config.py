"""Carga de configuración, semillas y utilidades de E/S compartidas."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_PKG_ROOT = Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """configs/default.yaml del directorio actual o, si no existe, del repositorio."""
    for base in (Path.cwd(), _PKG_ROOT):
        candidate = base / "configs" / "default.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No se encontró configs/default.yaml; pase --config.")


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Lee el YAML. Prioridad: argumento > variable ECGVLM_CONFIG > configs/default.yaml."""
    path = Path(path or os.environ.get("ECGVLM_CONFIG") or default_config_path())
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str(Path(v).expanduser()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="Ruta a un YAML de configuración.")
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No serializable: {type(o)}")


def manifest_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["work_dir"]) / "manifest.csv"


def load_manifest(cfg: dict):
    import pandas as pd

    path = manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute primero: python -m ecgvlm.preprocess")
    return pd.read_csv(path)
