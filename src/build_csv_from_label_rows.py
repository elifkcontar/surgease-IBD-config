import csv
import logging
import re

import hydra
from encord.client import EncordClientProject
from omegaconf import DictConfig
from tqdm import tqdm

from constants import CSV_FIELDS
from classes.project import Score 
from encord_utils.download import download_all_label_rows, download_all_videos
from utils import get_csv_path, set_encord_log_level

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    set_encord_log_level()

    project_client = EncordClientProject.initialise(
        cfg.project.project_hash, cfg.project.api_key
    )
    label_rows = download_all_label_rows(
        project_client, cache_dir=cfg.project.data_path
    )
    video_paths = download_all_videos(
        label_rows.values(), cache_dir=cfg.project.data_path, simulate=False
    )

    csv_file = get_csv_path(cfg)
    logger.info(f"Storing dataset information in {csv_file.absolute()}")

    score_fn: Score = hydra.utils.instantiate(cfg.project.score)

    with csv_file.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for label_hash, label_row in tqdm(label_rows.items(), "Building CSV"):
            data_unit = next((du for du in label_row["data_units"].values()))
            video_name = data_unit["data_title"]
            score_match = re.match(
                r"(video|UC)\s?(\d+)\s?-\s?(?P<score>\d+)(!=$|\.mp4)",
                video_name,
            )
            video_uceis_score = (
                int(score_match.group("score")) if score_match else -1
            )

            frame_images = video_paths.get(label_hash)

            if frame_images is None:
                continue

            classification_answers = label_row["classification_answers"]

            for frame in data_unit["labels"]:
                classifications = data_unit["labels"][str(frame)][
                    "classifications"
                ]

                try:
                    frame_path = frame_images.get(int(frame))
                except ValueError:
                    # Weird shit in label rows for the reference project.
                    continue

                if frame_path is None:
                    continue

                scores = {}
                for classification in classifications:
                    answer = classification_answers.get(
                        classification["classificationHash"]
                    )

                    if answer is None or len(answer) == 0:
                        continue

                    answer_classification = answer["classifications"][0]
                    if len(answer_classification) == 0:
                        continue

                    answers = answer_classification["answers"]
                    if len(answers) == 0:
                        continue

                    answer_hash = answers[0]["featureHash"]

                    score_result = score_fn(
                        classification["featureHash"], answer_hash
                    )
                    if score_result is None:
                        # missing_attribute = True
                        continue

                    sub_score, score_value = score_result
                    scores[sub_score.name] = score_value

                if len(scores) < 3:
                    continue

                uceis_score = sum(scores.values())

                writer.writerow(
                    {
                        "frame_path": frame_path.as_posix(),
                        "video_name": video_name,
                        "legacy_video_name": video_name.replace(
                            "video", "UC"
                        ).replace(" ", ""),
                        "frame": int(frame),
                        "video_uceis_score": video_uceis_score,
                        "vascular_score": scores["vascular"],
                        "bleeding_score": scores["bleeding"],
                        "erosion_score": scores["erosion"],
                        "UCEIS_score": uceis_score,
                    }
                )


if __name__ == "__main__":
    from pathlib import Path
    #Path('C:\\Dataset\\reference\\5b28f14b-5f15-4968-af8c-206080b508d5').mkdir(parents=True, exist_ok=True)
    main()
