"""Shared test doubles for the Ultralytics model.

Keeping these here means detector and tracker tests exercise the same stub, and
neither needs weights, a network, or a GPU.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from smartcity_vision.detection import detector as detector_module

COCO_SUBSET = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# (x1, y1, x2, y2, confidence, class_id)
BoxRow = tuple[float, float, float, float, float, int]


class FakeBoxes:
    """Minimal stand-in for ``ultralytics.engine.results.Boxes``.

    Values are torch tensors because the real class returns tensors and the
    detector calls ``.cpu()`` on them.
    """

    def __init__(self, rows: list[BoxRow], ids: list[int] | None = None) -> None:
        self.xyxy = torch.tensor([row[:4] for row in rows], dtype=torch.float32)
        self.conf = torch.tensor([row[4] for row in rows], dtype=torch.float32)
        self.cls = torch.tensor([row[5] for row in rows], dtype=torch.float32)
        # Ultralytics sets this to None until the tracker confirms a track.
        self.id = None if ids is None else torch.tensor(ids, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class FakeResult:
    """Minimal stand-in for ``ultralytics.engine.results.Results``."""

    def __init__(self, boxes: FakeBoxes | None) -> None:
        self.boxes = boxes


class FakeYolo:
    """Records calls and returns canned results for both predict and track."""

    def __init__(
        self,
        weights: str,
        rows: list[BoxRow] | None = None,
        raises: bool = False,
    ) -> None:
        self.weights = weights
        self.names = dict(COCO_SUBSET)
        self.predict_kwargs: list[dict[str, Any]] = []
        self.track_kwargs: list[dict[str, Any]] = []
        self._rows = rows if rows is not None else []
        self._raises = raises
        self._queued: list[tuple[list[BoxRow], list[int] | None]] = []
        self._call = 0

    def queue(self, rows: list[BoxRow], ids: list[int] | None) -> None:
        """Queue the boxes and identities the next ``track`` call should return."""
        self._queued.append((rows, ids))

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.predict_kwargs.append(kwargs)
        if self._raises:
            raise RuntimeError("CUDA out of memory")
        return [FakeResult(FakeBoxes(self._rows) if self._rows else None)]

    def track(self, **kwargs: Any) -> list[FakeResult]:
        self.track_kwargs.append(kwargs)
        if self._raises:
            raise RuntimeError("CUDA out of memory")
        rows, ids = self._queued[self._call]
        self._call += 1
        return [FakeResult(FakeBoxes(rows, ids) if rows else None)]


def install_fake_yolo(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[BoxRow] | None = None,
    raises: bool = False,
) -> list[FakeYolo]:
    """Replace the YOLO class with a stub; returns the list of instances built."""
    built: list[FakeYolo] = []

    def factory(weights: str) -> FakeYolo:
        model = FakeYolo(weights, rows=rows, raises=raises)
        built.append(model)
        return model

    monkeypatch.setattr(detector_module, "YOLO", factory)
    return built
