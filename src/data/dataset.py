import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.utils.class_weight import compute_class_weight
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class Label(Enum):
    VASCULAR = "vascular"
    EROSION = "erosion"
    BLEEDING = "bleeding"
    ALL = "all"


LABEL_ORDER = [Label.VASCULAR, Label.BLEEDING, Label.EROSION]


class SurgeaseDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        label: Label = Label.ALL,
        transform: Optional[Callable] = None,
        balance: bool = False,
    ):
        self.root: Path = root / split
        self.split = split
        self.label = label

        self.data = pd.read_csv((self.root / "labels.csv").as_posix())
        self._balance = balance
        self.balance()

        self.files = self.data[["file_name", "is_relative"]]
        if label == Label.ALL:
            self.labels = self.data[[l.value for l in LABEL_ORDER]].to_numpy()
        else:
            self.labels = self.data[label.value].to_numpy()

        self.transform = transform

    def balance(self):
        if not self._balance:
            return

        if self.label == Label.ALL:
            logger.info("Not balancing when Label is ALL")
            return
        else:
            label_values = self.data[self.label.value].to_numpy()
            labels, cnts = np.unique(label_values, return_counts=True)
            min_cnts = min(cnts)

            all_indices = set()
            for label in labels:
                indices = np.where(label_values == label)[0]
                indices = indices[
                    np.random.permutation(len(indices))[:min_cnts]
                ]
                all_indices = all_indices.union(set(indices))

            data_new = self.data.iloc[list(all_indices)]

            # Testing balancing
            label_values_new = self.data[self.label.value].to_numpy()
            labels_new, cnts_new = np.unique(label_values, return_counts=True)
            assert len(label_values_new) <= len(label_values)
            assert len(labels) == len(labels_new)
            assert len(cnts) == len(cnts_new)

            self.data = data_new

    def compute_class_weights(
        self, label: Optional[Label] = None, classes=None
    ):
        if self.label == Label.ALL:
            if label is None:
                raise ValueError(
                    "for Label.ALL, you need to specify which label to use."
                )

            labels = self.labels[:, LABEL_ORDER.index(label)]
        else:
            labels = self.labels

        classes = np.unique(labels)
        print(compute_class_weight("balanced", classes=classes, y=labels))
        return compute_class_weight("balanced", classes=classes, y=labels)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_image(img_path: Path, fmt: Optional[str] = None) -> Image:
        image = ImageOps.exif_transpose(Image.open(img_path))
        image.load()
        if fmt and not image.mode == fmt:
            image = image.convert(fmt)
        return image

    def __getitem__(self, idx):
        file_pth, is_relative = self.files.iloc[idx]
        if is_relative:
            fpth = self.root / self.files[idx]
        else:
            fpth = Path(file_pth)

        img = self.load_image(fpth)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_label_details(self, idx: int) -> Tuple[Dict, np.ndarray]:
        """
        :param idx: The sample index
        :return:
            - dict: A dict with DATASET_FIELD_NAMES key
            - ndarray: the label returned by __getitem__
        """
        if idx >= len(self.data) or idx < 0:
            raise ValueError(
                f"Index {idx} outside range [{0}:{len(self.data)}]"
            )

        return self.data.iloc[idx], self.labels[idx]
