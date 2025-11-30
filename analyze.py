#!/usr/bin/env python3
"""
Quick utility to pull the Minecraft DFoT weights and summarize their contents.
Run from the repo root: python scripts/inspect_minecraft_ckpt.py
"""
from pathlib import Path
import torch

from utils.ckpt_utils import download_pretrained, download_checkpoint

def fetch_minecraft_ckpt() -> Path:
    # Hugging Face shortcut used in the README/commands
    ckpt_path = download_pretrained("pretrained:DFoT_MCRAFT.ckpt")
    return Path(ckpt_path)

def summarize_ckpt(ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint from {ckpt_path}")
    print("Top-level checkpoint keys:", checkpoint.keys())

    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        print(f"State dict has {len(sd)} tensors; first 20 keys:")
        for i, key in enumerate(sd.keys()):
            if i == 20:
                print("...")
                break
            tensor = sd[key]
            shape = tuple(tensor.shape) if hasattr(tensor, "shape") else "scalar"
            print(f"  {key}: {shape}")
    else:
        print("No 'state_dict' key found; inspect manually.")

if __name__ == "__main__":
    ckpt_path = fetch_minecraft_ckpt()
    summarize_ckpt(ckpt_path)