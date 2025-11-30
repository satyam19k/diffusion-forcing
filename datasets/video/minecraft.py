from typing import Any, Dict, Optional, List
import io
import tarfile
from pathlib import Path

import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
)


class MinecraftBaseVideoDataset(BaseVideoDataset):
    _ALL_SPLITS = ["training", "validation"]

    def download_dataset(self):
        from internetarchive import download

        part_suffixes = [
            "aa",
            "ab",
            "ac",
            "ad",
            "ae",
            "af",
            "ag",
            "ah",
            "ai",
            "aj",
            "ak",
        ]

        # Download each part into self.save_dir / minecraft_marsh_dataset_xx
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        # Concatenate parts into one tar file on disk (no giant BytesIO in RAM)
        archive_path = self.save_dir / "minecraft.tar"
        with archive_path.open("wb") as combined:
            for part_suffix in part_suffixes:
                identifier = f"minecraft_marsh_dataset_{part_suffix}"
                file_name = f"minecraft.tar.part{part_suffix}"
                part_file = self.save_dir / identifier / file_name
                with part_file.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        combined.write(chunk)

        # Extract
        with tarfile.open(archive_path, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)

        # Move into training/validation
        minecraft_root = self.save_dir / "minecraft"
        (minecraft_root / "test").rename(self.save_dir / "validation")
        (minecraft_root / "train").rename(self.save_dir / "training")
        minecraft_root.rmdir()

        # Delete tar parts + big tar to save space
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_dir = self.save_dir / identifier
            part_file = part_dir / file_name
            if part_file.exists():
                part_file.unlink()
            if part_dir.exists():
                part_dir.rmdir()
        if archive_path.exists():
            archive_path.unlink()

    # *** NEW: auto-build metadata if missing ***
    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        For Minecraft, metadata is just a list of dicts with a 'video_paths' key
        pointing to each .mp4 file. If training.pt / validation.pt is missing,
        we scan the split folder and rebuild it.
        """
        metadata_dir = self.save_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = metadata_dir / f"{self.split}.pt"

        # If file exists, just load it
        if metadata_path.exists():
            return torch.load(metadata_path, map_location="cpu")

        # Otherwise, rebuild from filesystem
        split_root = self.save_dir / self.split
        video_paths = sorted(split_root.glob("*/*.mp4"))  # e.g. training/1_10/000000.mp4

        metadata: List[Dict[str, Any]] = []
        for vp in video_paths:
            metadata.append({"video_paths": vp})

        torch.save(metadata, metadata_path)
        print(f"[MinecraftBaseVideoDataset] Rebuilt metadata for split={self.split} "
              f"with {len(metadata)} videos at {metadata_path}")
        return metadata

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        # Each minecraft clip has exactly 300 frames.
        return 300

    def build_transform(self):
        return transforms.Resize(
            (self.resolution, self.resolution),
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )


class MinecraftSimpleVideoDataset(MinecraftBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        BaseSimpleVideoDataset.__init__(self, cfg, split)


class MinecraftAdvancedVideoDataset(
    MinecraftBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Minecraft advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "test":
            split = "validation"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"].with_suffix(".npz")
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(4)[actions]).float()
