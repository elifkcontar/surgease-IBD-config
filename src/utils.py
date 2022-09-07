import logging
from pathlib import Path

from omegaconf import DictConfig


def get_csv_path(cfg: DictConfig):
    data_dir = Path(cfg.project.data_path)
    csv_suffix = cfg.run.data_suffix
    if csv_suffix and csv_suffix[0] != "_":
        csv_suffix = f"_{csv_suffix}"
    csv_file = data_dir / f"data_rows{csv_suffix}.csv"
    return csv_file


def set_encord_log_level(level=logging.WARNING):
    for logger_name in logging.root.manager.loggerDict:
        if "encord" in logger_name:
            logging.getLogger(logger_name).setLevel(level)
