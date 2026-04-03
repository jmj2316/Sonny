"""Hugging Face Hub loader for Sonny weights."""

from __future__ import annotations

import json
from typing import Any

import torch

from networks.sonny import Sonny


def load_sonny_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    map_location: str | torch.device = "cpu",
) -> Sonny:
    """Load ``config.json`` + ``model.safetensors`` from the Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install huggingface_hub safetensors") from e

    cfg_path = hf_hub_download(repo_id, "config.json", revision=revision, token=token)
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)
    init_args = cfg.get("init_args")
    if not isinstance(init_args, dict):
        raise ValueError("config.json must contain 'init_args'")

    weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision, token=token)
    state = load_file(weights_path, device=str(map_location))
    model = Sonny(**init_args)
    model.load_state_dict(state, strict=True)
    return model
