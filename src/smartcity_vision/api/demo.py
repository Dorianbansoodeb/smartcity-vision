"""Live-demo job runner.

The public demo analyses the bundled sample clip on the server when a visitor
clicks Analyze. Only one job runs at a time; the HTTP handlers return immediately
and the browser polls for status. Annotated output is transcoded to H.264 so
browsers can play it (OpenCV's ``mp4v`` writer is not web-safe).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from smartcity_vision.cli import _persist_run, _write_summary
from smartcity_vision.detection.tracker import YoloTracker
from smartcity_vision.exceptions import SmartCityVisionError
from smartcity_vision.monitoring.metrics import record_run
from smartcity_vision.utils.config import AppConfig, load_config
from smartcity_vision.utils.logging import get_logger
from smartcity_vision.video.processor import VideoProcessor, create_source_from_config

logger = get_logger(__name__)

router = APIRouter()

SAMPLE_VIDEO = Path("data/input/traffic.mp4")
OUTPUT_DIR = Path("data/output")
WEB_VIDEO = OUTPUT_DIR / "demo_annotated.mp4"
DEMO_MAX_FRAMES = 64

JobStatus = Literal["idle", "running", "done", "error"]

_job_lock = threading.Lock()
_tracker_lock = threading.Lock()
_tracker: YoloTracker | None = None
_job: dict[str, Any] = {
    "status": "idle",
    "message": "Ready to analyse the sample clip.",
    "stats": None,
}


class DemoStatus(BaseModel):
    """Public job status for the demo UI."""

    status: JobStatus
    message: str
    stats: dict[str, Any] | None = None
    has_output: bool = False
    sample_ready: bool = False
    max_frames: int = DEMO_MAX_FRAMES


def _status_payload() -> DemoStatus:
    """Snapshot the in-memory job plus whether media files exist on disk."""
    return DemoStatus(
        status=_job["status"],
        message=_job["message"],
        stats=_job["stats"],
        has_output=WEB_VIDEO.is_file(),
        sample_ready=SAMPLE_VIDEO.is_file(),
        max_frames=DEMO_MAX_FRAMES,
    )


@router.get("/demo/status", response_model=DemoStatus)
def demo_status() -> DemoStatus:
    """Return the current demo job state."""
    return _status_payload()


@router.post("/demo/analyze", response_model=DemoStatus)
def start_demo_analysis() -> DemoStatus:
    """Start a sample-clip analysis if the worker is free."""
    if not SAMPLE_VIDEO.is_file():
        raise HTTPException(status_code=503, detail="Sample clip is not available on this server")
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An analysis is already running")
    _job.update(
        status="running",
        message="Loading the model and analysing the sample clip…",
        stats=None,
    )
    thread = threading.Thread(target=_run_demo_job, name="smartcity-demo", daemon=True)
    thread.start()
    return _status_payload()


@router.get("/demo/source")
def demo_source() -> FileResponse:
    """Serve the bundled sample clip (H.264, browser-playable)."""
    if not SAMPLE_VIDEO.is_file():
        raise HTTPException(status_code=404, detail="Sample clip not found")
    return FileResponse(SAMPLE_VIDEO, media_type="video/mp4", filename="traffic.mp4")


@router.get("/demo/output")
def demo_output() -> FileResponse:
    """Serve the last annotated result, if one exists."""
    if not WEB_VIDEO.is_file():
        raise HTTPException(
            status_code=404,
            detail="No annotated video yet — run an analysis first",
        )
    return FileResponse(WEB_VIDEO, media_type="video/mp4", filename="annotated.mp4")


@router.get("/demo/preview")
def demo_preview() -> FileResponse:
    """Serve the pre-rendered annotated GIF so the page is not empty on cold start."""
    path = _preview_gif()
    if path is None:
        raise HTTPException(status_code=404, detail="Preview GIF is not packaged on this server")
    return FileResponse(path, media_type="image/gif", filename="preview.gif")


def _preview_gif() -> Path | None:
    """Resolve the pre-rendered GIF from the package or the repo docs folder."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "static" / "preview.gif", Path("docs/images/demo.gif")):
        if candidate.is_file():
            return candidate
    return None


def _run_demo_job() -> None:
    """Worker thread: run the pipeline, transcode, then release the lock."""
    try:
        config = _demo_config()
        source = create_source_from_config(config)
        detector = _shared_tracker(config)
        stats = VideoProcessor(config, detector, source=source).run()
        _persist_run(config, stats)
        _write_summary(config, stats)
        record_run(stats.inference_ms_samples, stats.pipeline_fps, stats.detections_by_class)
        _transcode_for_web(config.output.directory / config.output.annotated_video_name, WEB_VIDEO)
        _job.update(
            status="done",
            message="Analysis finished. Numbers below are from this run.",
            stats=stats.as_dict(),
        )
    except SmartCityVisionError as exc:
        logger.error("Demo analysis failed: %s", exc)
        _job.update(status="error", message=str(exc), stats=None)
    except Exception as exc:  # noqa: BLE001 — surface to the UI, do not crash the API
        logger.exception("Demo analysis crashed")
        _job.update(status="error", message=f"Analysis failed: {exc}", stats=None)
    finally:
        _job_lock.release()


def _transcode_for_web(source: Path, destination: Path) -> None:
    """Convert OpenCV output to a browser-safe H.264 MP4 when ffmpeg is present."""
    if not source.is_file():
        raise SmartCityVisionError(f"Annotated video was not written: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found; serving the raw OpenCV file (may not play in browsers)")
        if source != destination:
            shutil.copy2(source, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not destination.is_file():
        raise SmartCityVisionError(
            f"Could not transcode annotated video: {completed.stderr[-400:]}"
        )
    logger.info("Wrote browser-playable video to %s", destination)


def _demo_config() -> AppConfig:
    """Config for the public demo: CPU, privacy on, capped frames, video on."""
    return load_config(
        updates={
            "video": {
                "source": str(SAMPLE_VIDEO),
                "max_frames": DEMO_MAX_FRAMES,
                "display": False,
            },
            "output": {
                "directory": OUTPUT_DIR,
                "write_annotated_video": True,
                "annotated_video_name": "annotated_video.mp4",
            },
            "model": {"device": "cpu"},
            "privacy": {"enabled": True},
        }
    )


def _shared_tracker(config: AppConfig) -> YoloTracker:
    """Reuse one loaded model across demo runs; reset ByteTrack state each time."""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = YoloTracker(config.model, config.tracking)
        else:
            model = _tracker._model
            if getattr(model, "predictor", None) is not None:
                model.predictor = None
        return _tracker


def warmup_model() -> None:
    """Load weights on boot when the demo image asks for it.

    Tests and local ``uvicorn`` leave this off so pytest never downloads a model.
    """
    flag = os.environ.get("SMARTCITY_DEMO_WARMUP", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return
    logger.info("Warming the demo model")
    _shared_tracker(_demo_config())
    logger.info("Demo model is ready")


__all__ = ["router", "warmup_model"]
