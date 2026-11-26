"""Seeds, environment capture, hashing, small IO helpers (Section 11)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import zlib

import numpy as np


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_hash(s: str) -> int:
    """Deterministic across processes and machines (unlike Python's hash())."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_dir: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown (not a git checkout)"


def capture_env() -> dict:
    import torch
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "kaggle_image": os.environ.get("KAGGLE_DOCKER_IMAGE", "n/a")}
    for mod in ("transformers", "tokenizers", "sklearn", "numpy", "pandas", "lime"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "not installed"
    try:
        env["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"],
                                                    stderr=subprocess.DEVNULL).decode().splitlines()
    except Exception:
        env["pip_freeze"] = []
    return env


def write_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
