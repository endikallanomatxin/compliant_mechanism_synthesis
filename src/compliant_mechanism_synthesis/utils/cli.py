from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch


def timestamped_run_dir(root: str | Path, name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = Path(root) / f"{timestamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_torch_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved
