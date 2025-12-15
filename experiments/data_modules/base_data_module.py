from typing import Dict
import os
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, root_cfg: DictConfig, compatible_datasets: Dict) -> None:
        super().__init__()
        self.root_cfg = root_cfg
        self.exp_cfg = root_cfg.experiment
        self.compatible_datasets = compatible_datasets

    # def _build_dataset(self, split: str) -> torch.utils.data.Dataset:
    #     if split in ["training", "test", "validation"]:
    #         return self.compatible_datasets[self.root_cfg.dataset._name](
    #             self.root_cfg.dataset, split=split
    #         )
    #     else:
    #         raise NotImplementedError(f"split '{split}' is not implemented")

    def _build_dataset(self, split: str) -> torch.utils.data.Dataset:
        if split in ["training", "test", "validation"]:
            # Fallback: if _name is not set, try to infer from config
            if not hasattr(self.root_cfg.dataset, '_name') or self.root_cfg.dataset._name is None:
                # Try to get from the config file name or other means
                # For now, raise a clearer error
                raise ValueError(
                    "dataset._name is not set! This should be set automatically by Hydra. "
                    "Make sure you're using 'dataset=minecraft' in your command."
                )
            
            dataset_name = self.root_cfg.dataset._name
            if dataset_name not in self.compatible_datasets:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found. Available: {list(self.compatible_datasets.keys())}"
                )
            
            return self.compatible_datasets[dataset_name](
                self.root_cfg.dataset, split=split
            )
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

    @staticmethod
    def _get_shuffle(dataset: torch.utils.data.Dataset, default: bool) -> bool:
        return not isinstance(dataset, torch.utils.data.IterableDataset) and default

    @staticmethod
    def _get_num_workers(num_workers: int) -> int:
        return min(os.cpu_count(), num_workers)

    def _dataloader(self, split: str) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        dataset = self._build_dataset(split)
        split_cfg = self.exp_cfg[split]
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=split_cfg.batch_size,
            num_workers=self._get_num_workers(split_cfg.data.num_workers),
            shuffle=self._get_shuffle(dataset, split_cfg.data.shuffle),
            persistent_workers=split == "training",
            worker_init_fn=lambda worker_id: (
                dataset.worker_init_fn(worker_id)
                if hasattr(dataset, "worker_init_fn")
                else None
            ),
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader("training")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("test")
