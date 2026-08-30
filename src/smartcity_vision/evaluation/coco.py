"""Download and slice COCO val2017 for a traffic-class evaluation.

The official val set is ~5k images. This module builds a deterministic 80-image
slice that actually contains the road-user classes we care about, then downloads
only those JPEGs. Full COCO val mAP is a different (and much larger) protocol;
do not compare the two.
"""

from __future__ import annotations

import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from smartcity_vision.evaluation.boxes import TruthBox
from smartcity_vision.exceptions import SmartCityVisionError
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
INSTANCES_NAME = "annotations/instances_val2017.json"

# COCO category names we evaluate. These match YOLOv8's COCO names.
DEFAULT_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)

# Quotas bias the slice toward vehicles instead of the person-heavy COCO prior.
_CLASS_QUOTA: dict[str, int] = {
    "person": 22,
    "car": 22,
    "bus": 10,
    "truck": 10,
    "bicycle": 8,
    "motorcycle": 8,
}


def coco_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert a COCO ``[x, y, w, h]`` box to ``(x1, y1, x2, y2)``."""
    x, y, width, height = bbox
    return (float(x), float(y), float(x + width), float(y + height))


def load_instances(path: Path) -> dict[str, Any]:
    """Read a COCO instances JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SmartCityVisionError(f"Cannot read COCO annotations: {path}") from exc
    if not isinstance(payload, dict) or "annotations" not in payload:
        raise SmartCityVisionError(f"{path} is not a COCO instances file")
    return payload


def ensure_instances(cache_dir: Path) -> Path:
    """Return ``instances_val2017.json``, downloading the official zip if needed."""
    extracted = cache_dir / "instances_val2017.json"
    if extracted.is_file():
        return extracted
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "annotations_trainval2017.zip"
    if not archive.is_file():
        logger.info("Downloading COCO annotations zip to %s", archive)
        _download(COCO_ANNOTATIONS_URL, archive)
    logger.info("Extracting %s", INSTANCES_NAME)
    try:
        with zipfile.ZipFile(archive) as zipped:
            payload = zipped.read(INSTANCES_NAME)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise SmartCityVisionError("COCO zip did not contain instances_val2017.json") from exc
    extracted.write_bytes(payload)
    return extracted


def select_slice(
    instances: dict[str, Any],
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    max_images: int = 80,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a compact COCO-format JSON for a deterministic class-balanced slice."""
    categories = {
        int(item["id"]): str(item["name"])
        for item in instances["categories"]
        if str(item["name"]) in classes
    }
    wanted_ids = set(categories)
    images = {int(item["id"]): item for item in instances["images"]}
    by_class: dict[str, list[int]] = defaultdict(list)
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in instances["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in wanted_ids:
            continue
        image_id = int(annotation["image_id"])
        if image_id not in images:
            continue
        anns_by_image[image_id].append(annotation)
        if int(annotation.get("iscrowd", 0)) == 0:
            by_class[categories[category_id]].append(image_id)

    rng = random.Random(seed)
    selected: list[int] = []
    used: set[int] = set()
    for class_name, quota in _CLASS_QUOTA.items():
        if class_name not in classes:
            continue
        candidates = list(dict.fromkeys(by_class.get(class_name, [])))
        rng.shuffle(candidates)
        taken = 0
        for image_id in candidates:
            if image_id in used:
                continue
            selected.append(image_id)
            used.add(image_id)
            taken += 1
            if taken >= quota or len(selected) >= max_images:
                break
        if len(selected) >= max_images:
            break

    leftover = [image_id for image_id in anns_by_image if image_id not in used]
    rng.shuffle(leftover)
    for image_id in leftover:
        if len(selected) >= max_images:
            break
        selected.append(image_id)

    selected_set = set(selected)
    return {
        "info": {
            "description": "SmartCity Vision COCO val2017 traffic-class slice",
            "seed": seed,
            "max_images": max_images,
            "source": "COCO 2017 val instances",
        },
        "images": [images[image_id] for image_id in selected],
        "annotations": [
            annotation for image_id in selected for annotation in anns_by_image[image_id]
        ],
        "categories": [item for item in instances["categories"] if int(item["id"]) in wanted_ids],
        "images_considered": len(selected_set),
    }


def truths_from_slice(slice_json: dict[str, Any]) -> list[TruthBox]:
    """Convert a slice JSON into :class:`TruthBox` records."""
    categories = {int(item["id"]): str(item["name"]) for item in slice_json["categories"]}
    boxes: list[TruthBox] = []
    for annotation in slice_json["annotations"]:
        name = categories.get(int(annotation["category_id"]))
        if name is None:
            continue
        boxes.append(
            TruthBox(
                image_id=_image_key(int(annotation["image_id"])),
                class_name=name,
                bbox=coco_xywh_to_xyxy(list(annotation["bbox"])),
                is_crowd=int(annotation.get("iscrowd", 0)) == 1,
            )
        )
    return boxes


def download_slice_images(slice_json: dict[str, Any], image_dir: Path) -> list[Path]:
    """Download each image in the slice. Existing files are reused."""
    image_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for image in slice_json["images"]:
        file_name = str(image["file_name"])
        destination = image_dir / file_name
        if not destination.is_file():
            url = COCO_IMAGE_URL.format(file_name=file_name)
            logger.info("Downloading %s", file_name)
            _download(url, destination)
        paths.append(destination)
    return paths


def image_id_from_path(path: Path) -> str:
    """Stable image id used to join predictions to COCO annotations."""
    return path.stem


def _image_key(coco_id: int) -> str:
    return f"{coco_id:012d}"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "smartcity-vision/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(destination)
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        raise SmartCityVisionError(f"Download failed: {url}") from exc


__all__ = [
    "COCO_ANNOTATIONS_URL",
    "DEFAULT_CLASSES",
    "download_slice_images",
    "ensure_instances",
    "image_id_from_path",
    "load_instances",
    "select_slice",
    "truths_from_slice",
]
