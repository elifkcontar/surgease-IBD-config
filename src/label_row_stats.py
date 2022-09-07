import logging
import os
from datetime import date, datetime
from pprint import pformat

import hydra
import matplotlib.pyplot as plt
import numpy as np
from encord.client import EncordClientProject
from omegaconf import DictConfig

from src.classes.project import Score
from src.encord_utils.download import download_all_label_rows
from utils import set_encord_log_level

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

    # Print stats
    logger.info("Analysing label rows.")
    object_counts = {}
    lr_colors = []

    cmaps = {
        "vascular": plt.get_cmap("summer"),
        "erosion": plt.get_cmap("autumn"),
        "bleeding": plt.get_cmap("winter"),
    }
    score_fn: Score = hydra.utils.instantiate(cfg.project.score)

    DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
    dates_changed = []

    for label_hash, label_row in label_rows.items():
        object_answers = label_row["object_answers"]
        classification_answers = label_row["classification_answers"]
        frame_colors = {"vascular": [], "erosion": [], "bleeding": []}
        lr_colors.append(frame_colors)

        for data_unit in label_row["data_units"].values():
            count_dict = object_counts.setdefault(
                "video_objects", {}
            )  # type: dict
            for frame, labels in data_unit["labels"].items():
                for obj in labels["objects"]:
                    name = obj["name"]
                    try:
                        classifications = object_answers[obj["objectHash"]][
                            "classifications"
                        ]
                        classification = classifications[0]
                        class_name = classification["name"]
                        answer = classification["answers"][0]["value"]
                        hash = classification["answers"][0]["featureHash"]
                        key = f"{class_name};{answer};{hash[:8]}"

                        sub_counts = count_dict.setdefault(
                            name, {}
                        )  # type: dict
                        sub_counts[key] = sub_counts.setdefault(key, 0) + 1
                    except (IndexError, KeyError):
                        sub_counts = count_dict.setdefault(
                            name, {}
                        )  # type: dict
                        sub_counts[";unclassified"] = (
                            sub_counts.setdefault(";unclassified", 0) + 1
                        )

            count_dict = object_counts.setdefault(
                "video_classifications", {}
            )  # type: dict
            frame_items = []
            for fr_name in data_unit["labels"]:
                try:
                    fr_int = int(fr_name)
                    frame_items.append((fr_int, data_unit["labels"][fr_name]))

                except ValueError:
                    pass
            frame_items = sorted(frame_items, key=lambda x: x[0])

            def get_color(feature_node_hash: str):
                if feature_node_hash is None:
                    return "#ffffff"

                feature_hash, severity = feature_node_hash.split("_")
                severity = int(severity) / 3
                color = cmaps.get(feature_hash)(severity)
                return color

            for frame, labels in frame_items:
                uceis_collect = {}

                for clf in labels["classifications"]:
                    name = clf["name"]
                    try:
                        dt = datetime.strptime(clf["createdAt"], DATE_FORMAT)
                        day = date(year=dt.year, month=dt.month, day=dt.day)
                        dates_changed.append(day)
                    except:
                        pass

                    classification_answer = classification_answers[
                        clf["classificationHash"]
                    ]
                    feature_hash = clf["featureHash"]
                    classification = classification_answer["classifications"][0]
                    class_name = classification["name"]
                    answer = classification["answers"][0]["value"]
                    option_hash = classification["answers"][0]["featureHash"]

                    score = score_fn(feature_hash, option_hash)

                    if score is None:
                        # Count other
                        sub_counts = count_dict.setdefault(
                            name, {}
                        )  # type: dict
                        sub_counts[";unclassified"] = (
                            sub_counts.setdefault(";unclassified", 0) + 1
                        )
                        uceis_collect["unknown"] = -1  # count unknowns as -1
                        continue

                    sub_count, score = score
                    uceis_collect[sub_count.name] = score

                    key = f"{class_name};{answer};{option_hash[:8]}"

                    sub_counts = count_dict.setdefault(name, {})  # type: dict
                    sub_counts[key] = sub_counts.setdefault(key, 0) + 1

                uceis_score = sum(uceis_collect.values())
                uceis_scores = count_dict.setdefault("uceis", {})
                uceis_scores[uceis_score] = (
                    uceis_scores.setdefault(uceis_score, 0) + 1
                )

                for key in frame_colors:
                    if key in uceis_collect:
                        score = uceis_collect[key]
                        color = get_color(f"{key}_{score}")
                    else:
                        color = get_color(None)

                    frame_colors[key].append(color)

    logger.info("Object classifications")
    logger.info(pformat(object_counts))

    project_title = project_client.get_project().title

    stats = object_counts["video_classifications"]
    fig, ax = plt.subplots(len(stats) - 1, 1, figsize=(10, 3 * len(stats) - 1))

    for (main_item, main_value), a in zip(stats.items(), ax):
        if main_item == "Skipped frame?":
            continue

        keys, values = np.array(list(main_value.items())).T
        values = values.astype(int)

        if isinstance(keys[0], str):
            keys = [k.split(";")[1] for k in keys]

        a.set_title(main_item)
        bars = a.bar(keys, values)
        a.bar_label(bars)

    fig.suptitle(f"{cfg.project.short_name}: {project_title}")
    fig.tight_layout()
    fig.savefig(f"{cfg.project.short_name}_classification_histogram.pdf")

    fig, ax = plt.subplots(figsize=(10, 10))
    max_rows = min(25, len(lr_colors))
    keys = sorted(list(lr_colors[0].keys()))

    for num, lr_color in enumerate(lr_colors[:max_rows]):
        for j, key in enumerate(keys):
            colors = lr_color[key]
            X = [i for i in range(len(colors)) if colors[i] != "#ffffff"]
            Y = [max_rows - num - j / (len(keys) + 2)] * len(X)
            colors = [c for c in colors if c != "#ffffff"]
            ax.scatter(X, Y, c=colors, s=8)

    fig.suptitle(f"{cfg.project.short_name}: Classification outline")
    fig.tight_layout()
    fig.savefig(f"{cfg.project.short_name}_classification_outline.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    _, _, bars = ax.hist(dates_changed)
    ax.bar_label(bars)

    fig.suptitle(f"{cfg.project.short_name}: Dates where label rows changed.")
    fig.tight_layout()
    fig.savefig(f"{cfg.project.short_name}_dates_changed.pdf")

    plt.show()

    logger.info(f"The plots are stored in {os.getcwd()}")


if __name__ == "__main__":
    main()
