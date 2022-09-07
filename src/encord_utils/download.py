import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import encord.exceptions
import requests
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)
_DATA_CACHE_DIR = Path("../../data/label_rows")


@dataclass
class DataSpecs:
    img: Path
    du: dict
    lr: dict


def collect_async(fn, job_args, key_fn, max_workers=4, **kwargs):
    """
    Distribute work across multiple workers. Good for, e.g., downloading data.
    Will return results in dictionary.
    :param fn: The function to be applied
    :param job_args: Arguments to `fn`.
    :param key_fn: Function to determine dictionary key for the result (given the same input as `fn`).
    :param max_workers: Number of workers to distribute work over.
    :param kwargs: Arguments passed on to tqdm.
    :return: Dictionary {key_fn(*job_args): fn(*job_args)}
    """
    job_args = list(job_args)
    if not isinstance(job_args[0], tuple):
        job_args = [(j,) for j in job_args]

    results = {}
    with tqdm(total=len(job_args), **kwargs) as pbar:
        with Executor(max_workers=max_workers) as exe:
            jobs = {exe.submit(fn, *args): key_fn(*args) for args in job_args}
            for job in as_completed(jobs):
                key = jobs[job]

                result = job.result()
                if result:
                    results[key] = result

                pbar.update(1)
    return results


def download_file(
    url: str,
    destination: Path,
    byte_size=1024,
):
    if destination.is_file():
        return destination

    r = requests.get(url, stream=True)
    with destination.open("wb") as f:
        for chunk in r.iter_content(chunk_size=byte_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return destination


def get_label_row(lr, client, cache_dir, refresh=False):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    if not lr["label_hash"]:
        return None

    cache_pth = cache_dir / lr["label_hash"] / "label_row.json"

    # Upgrade from legacy cache structure to have separate directory for each project.
    legacy_cache_path = cache_dir.parent / lr["label_hash"] / "label_row.json"
    if (not cache_pth.is_file()) and legacy_cache_path.is_file():
        cache_pth.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(legacy_cache_path, cache_pth)

    if not refresh and cache_pth.is_file():
        # Load cached version
        with cache_pth.open("r") as f:
            return json.load(f)

    else:
        try:
            lr = client.get_label_row(lr["label_hash"])
        except encord.exceptions.UnknownException as e:
            logger.warning(
                f"Failed to download label row with label_hash {lr['label_hash'][:8]}... and data_title {lr['data_title']}"
            )
            return None

        # Cache label row.
        cache_pth.parent.mkdir(parents=True, exist_ok=True)
        with cache_pth.open("w") as f:
            json.dump(lr, f, indent=2)

        return lr


def download_all_label_rows(client, **kwargs):
    return collect_async(
        partial(get_label_row, client=client, **kwargs),
        client.get_project().label_rows,
        lambda lr: lr["label_hash"],
        desc="Colleting label rows from Encord SDK.",
    )


def download_video_from_label_row(
    lr, cache_dir=_DATA_CACHE_DIR, simulate=False
):
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    label_hash = lr["label_hash"]

    if label_hash is None:
        return None

    label_pth = cache_dir / label_hash
    label_pth.mkdir(parents=True, exist_ok=True)

    lr_path = label_pth / "label_row.json"
    if not lr_path.exists():
        with (label_pth / "label_row.json").open("w") as f:
            json.dump(lr, f, indent=2)

    frame_pth = label_pth / "images"
    if (
        frame_pth.exists()
        and frame_pth.is_dir()
        and len(list(frame_pth.glob("*.jpg"))) > 0
    ):
        return {
            int(f.stem): f for f in frame_pth.iterdir() if f.suffix == ".jpg"
        }

    frame_pth.mkdir(parents=True, exist_ok=True)

    data_unit = next((du for du in lr["data_units"].values()))

    frames = set(
        map(int, filter(lambda x: re.match("\d", x), data_unit["labels"]))
    )

    frame_pths = {}
    if simulate:
        # Used mainly for debugging purposes.
        # NB: The returned file paths may not exist.
        return {
            frame_idx: frame_pth / f"{frame_idx:05d}.jpg"
            for frame_idx in frames
        }
    else:
        cap = cv2.VideoCapture(data_unit["data_link"])
        frame_idx = -1
        while cap.isOpened():
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx not in frames:
                continue

            img_path = frame_pth / f"{frame_idx:05d}.jpg"

            # TODO: Elif maybe downscale.
            cv2.imwrite(img_path.absolute().as_posix(), frame)
            frame_pths[frame_idx] = img_path

        return frame_pths


def download_all_videos(label_rows, **kwargs):
    return collect_async(
        partial(download_video_from_label_row, **kwargs),
        label_rows,
        lambda lr: lr["label_hash"],
        desc="Colleting frames from label rows.",
    )
