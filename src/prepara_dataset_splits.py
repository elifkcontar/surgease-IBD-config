import csv
import logging
import shutil
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from constants import CSV_FIELDS, DATASET_FIELD_NAMES
from utils import get_csv_path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    csv_file = get_csv_path(cfg)
    try:
        data = pd.read_csv(csv_file, names=CSV_FIELDS)
    except FileNotFoundError:
        raise ValueError(
            f"There does not seem to be a csv file for project {cfg.project.short_name}\n Run the `build_csv_from_label_rows.py` file first."
        )

    out_dir = Path(cfg.split.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    skip_cnt = 0

    if cfg.dataset.project != cfg.project.short_name:
        raise ValueError(
            f"Specified dataset `{cfg.dataset.name}` with project `{cfg.dataset.project}` does not match specified project short name `{cfg.project.short_name}`"
        )

    dataset = hydra.utils.instantiate(cfg.dataset)

    for split in dataset.splits:
        label_cnt = 0

        split_pth = out_dir / split.name
        split_pth.mkdir(exist_ok=True)

        label_file = split_pth / "labels.csv"
        with label_file.open("w", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=DATASET_FIELD_NAMES, dialect="excel"
            )
            writer.writeheader()

            videos_not_included = []

            
            for fname in tqdm(split.files, desc=f"Processing {split.name}"):
                #subset = data
                
                subset = data[data["video_name"] == fname]

                if len(subset) == 0:
                    subset = data[
                        data["legacy_video_name"].str.contains(
                            fname.split(".")[0]
                        )
                    ]

                    if len(subset) == 0:
                        videos_not_included.append(fname)
                
                for idx, row in subset.iterrows():
                    pth_from = Path(row["frame_path"])

                    if not pth_from.is_file():
                        skip_cnt += 1
                        # logger.warning(f"({skip_cnt}) Skipping pth_from")
                        continue

                    if pth_from.is_file() and cfg.split.copy_frames:
                        pth_to = split_pth / f"{label_cnt:06d}.jpg"
                        shutil.copy(pth_from, pth_to)
                        out_pth = pth_to
                        is_relative = True
                    else:
                        out_pth = pth_from.absolute()
                        is_relative = False

                    writer.writerow(
                        {
                            "file_name": out_pth,
                            "is_relative": is_relative,
                            "vascular": row["vascular_score"],
                            "erosion": row["bleeding_score"],
                            "bleeding": row["erosion_score"],
                            "uceis": row["UCEIS_score"],
                            "video_score": row["video_uceis_score"],
                        }
                    )
                    label_cnt += 1

                f.flush()
            if videos_not_included:
                print()  # print to force logging output to new line
                logger.warning(
                    f"Didn't include any samples from these videos: {videos_not_included}"
                )


if __name__ == "__main__":
    main()
