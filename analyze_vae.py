#!/usr/bin/env python3
from pathlib import Path
import torch
from utils.ckpt_utils import download_pretrained

def fetch_minecraft_vae_ckpt() -> Path:
    # This uses the same HF shortcut mechanism as DFoT_MCRAFT
    ckpt_path = download_pretrained("pretrained:ImageVAE_MCRAFT.ckpt")
    return Path(ckpt_path)

def summarize_ckpt(ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint from {ckpt_path}")
    print("Top-level keys:", ckpt.keys())

    sd = ckpt["state_dict"]
    # Show only VAE-related keys
    for k in list(sd.keys())[:40]:
        print(k, sd[k].shape)

    print("\nEncoder-related keys (filtering for 'encoder'):")
    for k in sd.keys():
        if "encoder" in k:
            print("  ", k)

if __name__ == "__main__":
    ckpt_path = fetch_minecraft_vae_ckpt()
    summarize_ckpt(ckpt_path)
