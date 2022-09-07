from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    files: List[str]


@dataclass
class Dataset:
    name: str
    description: str
    project: str
    splits: List[DatasetSplit]
