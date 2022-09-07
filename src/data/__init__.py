from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from torch.utils.data import DataLoader

from classes.dataset import Dataset, DatasetSplit

from .dataset import LABEL_ORDER, Label, SurgeaseDataset
from .transforms import get_transforms


@dataclass(frozen=True)
class DataWrapper:
    dataset: SurgeaseDataset
    loader: DataLoader
    split: DatasetSplit


def get_data(cfg) -> List[DataWrapper]:
    out = []
    train_transform, val_transform = get_transforms()
    cfg_dataset: Dataset = hydra.utils.instantiate(cfg.dataset)
    data_dir = Path(cfg.split.output_dir)

    for split in cfg_dataset.splits:
        train = "train" in split.name.lower()
        ds = SurgeaseDataset(
            root=data_dir,
            split=split.name,
            label=Label.ALL,
            transform=train_transform if train else val_transform,
            balance=False,
        )
        dl = DataLoader(
            ds, batch_size=cfg.model.batch_size, num_workers=4, pin_memory=True, shuffle=train
        )
        print('len of dataloader is')
        print(len(dl))
        out.append(DataWrapper(dataset=ds, loader=dl, split=split))

    return out
