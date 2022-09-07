"""
PRINT AND PLOT Label LOG DATES.
"""
import logging
from datetime import date, datetime, time
from functools import reduce
from time import mktime
from typing import Dict, List

import encord.exceptions
import hydra
import matplotlib.pyplot as plt
from encord.client import EncordClient
from encord.orm.label_log import Action, LabelLog
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    for lname in ["encord.configs", "encord.http.querier", "encord.client"]:
        logging.getLogger(lname).setLevel(logging.WARNING)

    project_client = EncordClient.initialise(
        cfg.project.project_hash, cfg.project.api_key
    )

    unix_ts = mktime(date(year=2021, month=6, day=1).timetuple())
    try:
        logs = project_client.get_label_logs(from_unix_seconds=unix_ts)
    except encord.exceptions.OperationNotAllowed:
        logger.error(
            "The API KEY does not have access to reading the label logs. Terminating."
        )
        exit(0)

    # Find overall latest
    parse_time = lambda x: datetime.strptime(
        x.created_at.split()[0], "%Y-%m-%d"
    )
    latest_time = reduce(
        lambda a, b: a if parse_time(a) > parse_time(b) else b, logs
    )
    print(f"Latest overall time is: {latest_time}")

    # Find latest add
    adds: List[LabelLog] = list(filter(lambda x: x.action == Action.ADD, logs))
    latest_add = reduce(
        lambda a, b: a if parse_time(a) > parse_time(b) else b, adds
    )
    print(f"Latest add is: {latest_add.created_at}")

    # Find latest start
    start: List[LabelLog] = list(
        filter(lambda x: x.action == Action.START, logs)
    )
    latest_start = reduce(lambda a, b: a if tm(a) > tm(b) else b, start)
    print(f"Latest start is: {latest_start.created_at}")

    # Find latest end
    end: List[LabelLog] = list(filter(lambda x: x.action == Action.END, logs))
    latest_end = reduce(lambda a, b: a if tm(a) > tm(b) else b, end)
    print(f"Latest end is: {latest_end.created_at}")

    # Find latest delete
    deletes: List[LabelLog] = list(
        filter(lambda x: x.action == Action.DELETE, logs)
    )
    latest_delete = reduce(lambda a, b: a if tm(a) > tm(b) else b, deletes)
    print(f"Latest delete is: {latest_delete.created_at}")

    # Plot stats
    delete_stats: Dict[str, List[date]] = {}
    max_time = datetime(year=2020, month=1, day=1)
    for label_log in deletes:
        t = datetime.strptime(label_log.created_at.split()[0], "%Y-%m-%d")
        delete_stats.setdefault(label_log.label_name, []).append(t)
        max_time = max(t, max_time)

    fig, ax = plt.subplots(len(delete_stats), figsize=(12, 20))
    for (k, v), a in zip(delete_stats.items(), ax):
        a.hist(v)
        a.set_title(k)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
