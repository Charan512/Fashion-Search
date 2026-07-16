"""
Configuration utilities for the Fashion Retrieval System.

Handles loading YAML config, expanding environment variables,
and providing typed access to configuration values.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env at import time so os.environ picks up keys
load_dotenv()
_ROOT = Path(__file__).resolve().parents[2]  # project root


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dict.

    Expands ``${VAR}`` placeholders in string values using ``os.environ``.

    Args:
        config_path: Absolute or relative path to ``config.yaml``.
                     Defaults to ``<project_root>/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is malformed.
    """
    path = Path(config_path) if config_path else _ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    config = _expand_env_vars(raw)
    logger.debug("Loaded config from %s", path)
    return config


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders in config values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """Retrieve an environment variable, with an optional default.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is absent.

    Returns:
        The variable's value or ``default``.

    Raises:
        EnvironmentError: If variable is absent and no default provided.
    """
    value = os.environ.get(name, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


def get_device(config: Optional[Dict] = None) -> str:
    """Resolve the compute device to use.

    Checks (in order): env var ``DEVICE``, config file, then auto-detects.

    Args:
        config: Optional loaded config dict.

    Returns:
        ``"cuda"`` if a GPU is available, else ``"cpu"``.
    """
    import torch

    device_env = os.environ.get("DEVICE", "")
    if device_env:
        return device_env

    if config:
        cfg_device = config.get("models", {}).get("clip", {}).get("device", "")
        if cfg_device:
            return cfg_device if (cfg_device == "cpu" or torch.cuda.is_available()) else "cpu"

    detected = "cuda" if torch.cuda.is_available() else "cpu"
    if detected == "cpu":
        logger.warning("No CUDA GPU detected — running on CPU (indexing will be slow).")
    return detected


def configure_logging(config: Optional[Dict] = None) -> None:
    """Set up root logger using config or defaults.

    Args:
        config: Optional loaded config dict.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    if config:
        log_level = config.get("logging", {}).get("level", log_level)

    log_format = (
        config.get("logging", {}).get("format", "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        if config
        else "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
    )
