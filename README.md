# SmartCity Vision

Turns traffic-camera video into structured traffic analytics with YOLOv8 and OpenCV.

The project is being built in phases, and the goal is a system that behaves like production
infrastructure rather than a detection demo: typed configuration, a video-source abstraction that
treats files, webcams, and RTSP streams identically, measured performance instead of estimated
performance, and tests that assert behaviour rather than existence.

**Status: Phase 2 of 12 complete** — configuration, video ingestion, YOLOv8 detection, ByteTrack /
BoT-SORT tracking, unique object counting, overlay rendering, an annotated-video writer, and a
measured run summary. Line/zone analytics, persistence, the API, and the MLOps layers land in later
phases (see [Roadmap](#roadmap)).

Every number in this README was produced by running the code on this machine. Nothing is estimated.

![Annotated frame showing a tracked car with a persistent ID](docs/images/phase2_tracked_frame.png)

## Quickstart

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Dorianbansoodeb/smartcity-vision.git
cd smartcity-vision

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt

# Fetch a public sample clip (people, bicycles, and cars in a car park).
curl -L -o data/input/traffic.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4

python scripts/run_video.py --input data/input/traffic.mp4
```

The first run downloads `yolov8n.pt` (6.2 MB) and caches it at `models/yolov8n.pt`, so later runs
work offline. Outputs land in `data/output/`:

| File | Contents |
| --- | --- |
| `annotated_video.mp4` | Source video with boxes, labels, confidences, track IDs, and a status panel |
| `run_summary.json` | Measured statistics, unique object counts, and the full config snapshot |

## Unique counting, and why the numbers differ

On the 647-frame sample clip the system reports **344 detections but 10 unique objects**
(2 bicycles, 3 cars, 5 people). Both numbers are correct and they answer different questions:
344 is how many boxes were drawn, 10 is how many distinct road users appeared. Reporting the first
as a traffic count is the single most common way this kind of project produces nonsense, so counting
is keyed on track identity and never on detections.

Three rules make the total defensible rather than merely plausible:

1. **One track counts once**, regardless of how many frames it appears in.
2. **A track must be seen `min_track_frames` times** (default 3) before it counts, so a one-frame
   false positive cannot inflate the total. On this clip that filtered exactly one track, which
   appeared in a single frame.
3. **A track's class is decided by majority vote** across its observations, because YOLO flickers
   between visually similar classes on the same object. If the majority later changes, the count is
   *moved* rather than duplicated.

Rule 3 is not hypothetical here. Measured track-by-track on this clip, one car was classified as
`car` in 47 frames and `bus` in 2, so a naive per-frame counter would report a phantom bus. Majority
vote attributes the track to `car` and the total stays at one object.

### Measured track persistence

The Phase 2 acceptance question is whether track IDs actually persist. Measured across the clip:

| Metric | Value |
| --- | --- |
| Distinct tracks | 11 |
| Tracked detections | 327 |
| Frames per track | min 1, mean 29.7, max 68 |
| Continuity (frames seen ÷ frames spanned) | 0.98 mean |

A continuity of 0.98 means IDs survive essentially unbroken over each object's time on screen rather
than being reassigned every few frames; the longest-lived track held one identity for 68 consecutive
frames. Counts were also bit-for-bit reproducible: all six tracking runs below reported the same 10
objects with the same class breakdown.

Note that 10 is *consistent*, not *validated*. The clip has no ground-truth labels, so this is a
plausibility and reproducibility result, not an accuracy result.

## Measured performance

Absolute latency on a fanless laptop drifts substantially with thermal state, so numbers are only
compared *within* a batch, each batch is 3 repeats, and the observed range is always shown next to
the median. Comparing a number from one batch against a number from another would be misleading.

Clip: 647 frames, 768×432, 12 fps (54 s of footage).
Hardware: Apple M3 (8 cores), macOS 26.5.1, Python 3.13.14, torch 2.13.0, ultralytics 8.4.135,
OpenCV 5.0.0, weights `yolov8n.pt` at `imgsz=640`, `conf=0.25`, `iou=0.45`, video writing disabled
so every configuration does identical work.

**Batch A — device comparison** (detection only, no tracking):

| Device | Avg inference | p95 inference | Throughput | Detections |
| --- | --- | --- | --- | --- |
| MPS (what `auto` picks here) | 16.93 ms [16.57–18.04] | 25.22 ms [23.65–26.30] | 55.34 fps [51.87–56.59] | 369 |
| CPU | 22.32 ms [22.04–22.59] | 25.65 ms [25.21–26.28] | 43.74 fps [43.18–44.28] | 369 |

MPS is about 24% faster than CPU on mean inference, but the two are within noise of each other at
the p95, and MPS is the more variable of the two. On this clip and this model size the accelerator
does not improve tail latency, which is the number that would matter for a real-time deployment.

**Batch B — tracker comparison** (MPS, configurations interleaved round-robin across repeats so
thermal drift is spread across all three rather than confounded with the tracker):

| Configuration | Avg inference | p95 inference | Throughput | Detections | Unique |
| --- | --- | --- | --- | --- | --- |
| Detection only | 18.85 ms [17.80–31.78] | 27.57 ms [26.23–52.24] | 49.45 fps [28.94–52.18] | 369 | n/a |
| ByteTrack | 19.65 ms [18.31–25.44] | 32.75 ms [28.10–47.92] | 47.15 fps [35.99–50.63] | 344 | 10 |
| BoT-SORT | 20.96 ms [20.41–21.34] | 31.94 ms [31.71–33.08] | 44.70 fps [43.98–46.00] | 345 | 10 |

Tracking is cheap: ByteTrack adds roughly 1 ms per frame over plain detection and BoT-SORT roughly
2 ms, so identity is not what limits throughput here. Both trackers produced identical unique counts,
so on this clip BoT-SORT's extra appearance modelling buys nothing measurable while costing latency
— which is why ByteTrack is the default.

The wide ranges in Batch B are honest and worth explaining: the *first* repeat of the batch was a
systematic outlier for every configuration (detection-only measured 31.78 ms, versus 17.80 ms by the
third repeat). Cold caches and thermal state, not the tracker, cause that. This is exactly why the
tables report medians over repeats rather than a single run, and why an earlier single-run
measurement here appeared to show tracking halving throughput — it did not.

Tracking also *reduces* raw detection count slightly (369 → 344), because the tracker discards
low-confidence detections it cannot associate with a track. That is a real behavioural difference
between the two modes, not a bug.

Latency is measured after an explicit warmup inference, because the first forward pass on MPS took
742–1425 ms across four measured runs — roughly 45–85× the steady state. Charging that to the first
real frame inflated the average enough to push it above the p95, which is exactly how misleading
benchmark numbers get published. `model.warmup` controls this.

An ONNX Runtime comparison against these PyTorch baselines is Phase 11; no ONNX numbers are claimed
yet.

## Command line

`scripts/run_video.py`, `scripts/run_webcam.py`, and `python -m smartcity_vision` share one
implementation in `src/smartcity_vision/cli.py`. The scripts work from a source checkout without
installing; `python -m smartcity_vision` and the `smartcity-vision` console script require
`pip install -e .`.

```bash
python scripts/run_video.py --input data/input/traffic.mp4      # file
python scripts/run_webcam.py --display                          # webcam 0 with a preview window
python scripts/run_video.py --input rtsp://camera.local/stream1  # network stream
python scripts/run_video.py --tracker botsort.yaml               # swap tracking algorithm
python scripts/run_video.py --no-tracking                        # stateless detection, no counting
```

Useful flags: `--device {auto,cpu,cuda,mps}`, `--conf`, `--iou`, `--imgsz`, `--tracker`,
`--no-tracking`, `--frame-skip`, `--max-frames`, `--loop`, `--display`, `--output-dir`, `--no-video`,
`--log-level`, and `--set key=value` for any config key without a dedicated flag:

```bash
python scripts/run_video.py --frame-skip 2 --max-frames 100 \
  --set analytics.counting.min_track_frames=5 \
  --set visualization.hud_position=top-right
```

## Configuration

All tunables live in [`config/default.yaml`](config/default.yaml) and are validated by Pydantic
models in `src/smartcity_vision/utils/config.py`. Precedence is defaults → YAML → `--set`
overrides → explicit CLI flags.

Validation is strict on purpose: unknown keys are rejected rather than ignored (a typo like
`confidenc_threshold` fails loudly instead of silently doing nothing), thresholds are range-checked,
and `image_size` must be a multiple of the YOLOv8 stride of 32 so it cannot be silently rounded.
Config sections are added as the phase that consumes them lands, so the file never contains options
that no code reads. Current sections: `model`, `tracking`, `analytics.counting`, `video`, `output`,
`visualization`, `logging`.

`AppConfig.snapshot()` produces the JSON config snapshot embedded in every `run_summary.json`. That
snapshot is the seed of the audit trail that Phase 5 turns into a `model_predictions_audit` table.

## Layout

```
config/default.yaml              Validated configuration
scripts/run_video.py             CLI entry points (thin wrappers over cli.py)
scripts/run_webcam.py
src/smartcity_vision/
├── cli.py                       Argument parsing and run orchestration
├── exceptions.py                Package error hierarchy
├── detection/detector.py        YoloDetector, Detection, DetectionResult, device resolution
├── detection/tracker.py         YoloTracker: identity assignment via ByteTrack/BoT-SORT
├── analytics/counter.py         UniqueObjectCounter, CountingSummary
├── video/source.py              VideoSource abstraction: file, webcam, stream + factory
├── video/processor.py           Frame pipeline and ProcessingStats
├── visualization/renderer.py    Boxes, labels, track IDs, status panel
└── utils/{config,logging}.py    Typed config and logging setup
tests/                           90 behavioural tests
```

Four boundaries matter for later phases. The video source yields plain `Frame` objects, so nothing
downstream knows whether frames came from a file or an RTSP camera. The detector converts Ultralytics
results into `Detection` objects immediately, so no other module depends on the Ultralytics API —
which is what makes the Phase 11 ONNX swap a contained change. `YoloTracker` subclasses
`YoloDetector` and overrides only the model call, so it returns the same `DetectionResult` type and
drops into the pipeline unchanged; the only difference is that detections carry a `track_id`. And
`VideoProcessor` is the only place that knows the order of operations, so line crossing, zones, and
density plug in there rather than into the detector or the renderer.

## Testing

```bash
pytest          # 90 tests, ~2 s, no weights or GPU required
ruff check .
ruff format --check .
```

The tests assert behaviour, not the existence of functions. Highlights:

- **Counting:** a track spanning 50 frames counts once; a track that disappears and returns is not
  recounted; a track below the confirmation threshold does not count yet; class flicker resolves by
  majority without double counting; a sustained class change *moves* the count instead of adding
  one; per-class counts always sum to the total; per-track memory stays bounded under 200 tracks.
- **Tracking:** IDs are attached and persist across frames; `persist=True` and the tracker name
  reach the model; detection thresholds still apply through the tracking call; unconfirmed tracks
  yield detections with `track_id=None`; warmup uses plain prediction so it cannot pollute tracker
  state.
- **Video:** decoded frames are checked for order, content, and frame-rate-derived timestamps against
  a video generated in a fixture; the factory dispatches webcam indices, URL schemes, and paths, and
  fails loudly on a missing file.
- **Detection:** result conversion, class-name-to-id filtering, single model load across frames, and
  error wrapping are tested against a stubbed Ultralytics model.
- **Pipeline:** frame skipping, frame limits, warmup exclusion, and non-mutation of the decoded frame
  are tested end to end with a stub detector.
- **Rendering:** pixel-level checks that a box outline is drawn without filling its interior, that a
  label on a corner box stays inside the frame, that the status panel lands in the configured corner,
  and that track IDs appear in labels only when present and enabled.

## Design notes

**Device selection.** `auto` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit request for
unavailable hardware logs a warning and falls back to CPU rather than crashing part-way through a
run.

**Tracking state.** Ultralytics keeps tracker state inside the model instance and requires
`persist=True` plus one in-order call per frame; without it every frame restarts the tracker and
every object gets a new ID. That is why the model is loaded once and reused, and why the warmup pass
deliberately uses plain prediction so a synthetic blank frame never enters tracker state.

**Bounded memory.** Per-track vote tallies are evicted once a track has been missing for
`forget_track_after_frames`, so a long or live stream does not accumulate state indefinitely. The set
of track identities and their attributed class is deliberately retained for the run, because that is
precisely what guarantees a reappearing ID is not counted twice.

**Weights handling.** A missing local weights file is treated as a request for the equivalently
named official checkpoint, downloaded straight to the configured path so it stays inside the project
rather than landing in the working directory, and so subsequent runs are offline and reproducible.

**Frame skipping.** `--frame-skip N` drops N frames between processed frames, and the annotated
output's frame rate is divided by the same stride so playback duration still matches the source.
Note that skipping frames also weakens tracking, since the tracker sees larger jumps between
observations.

**Timestamps.** File sources derive timestamps from the frame index and frame rate, which is
deterministic and reproducible. Live sources use a monotonic clock, because a stream has no
meaningful "frame N of M".

**Fail fast.** The video source is constructed before model weights are loaded, so a typo in a path
or an unreachable stream fails in milliseconds instead of after a multi-second model load.

**Interruption.** `Ctrl-C` finalises the video writer, writes the summary for the frames already
processed, and exits 130, instead of leaving a corrupt output file. A handled configuration or video
error exits 1 with a single readable message rather than a traceback.

## Known limitations after Phase 2

- **Counts are unvalidated.** The 10 unique objects are reproducible and plausible but the clip has
  no ground-truth labels, so no precision, recall, or mAP is reported. None has been measured, and
  none is invented here.
- **Counting is whole-frame.** There are no counting lines or zones yet, so the system answers "how
  many distinct road users appeared" and not "how many crossed this line, in which direction". That
  is Phase 3.
- **Identity is not guaranteed across occlusion.** A vehicle that is fully hidden and reappears may
  be issued a new ID and counted twice. BoT-SORT exists to mitigate this and is one flag away, but on
  this clip it produced no measurable difference — a harder clip would be needed to show its value.
- **Pretrained COCO weights, unvalidated on this domain.** Low light, rain, heavy occlusion, and
  unusual camera angles are untested. `MODEL_CARD.md` (Phase 12) will document this properly.
- **Single machine, thermally limited.** All figures come from one Apple M3 laptop; treat
  cross-batch comparisons as meaningless and within-batch comparisons as indicative only.
- **No privacy layer yet.** The annotated video persists identifiable pixels. Face and licence-plate
  anonymisation is Phase 7 and will default to on.
- **`mp4v` codec** is used for output because it is available in stock OpenCV wheels; it is not the
  most efficient choice and is configurable via `output.video_codec`.
- **Preview windows** need a desktop session; `--display` degrades to headless with a warning when
  no window can be opened.

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Config, video ingestion, YOLOv8 detection, rendering, measured runs | Complete |
| 2 | ByteTrack/BoT-SORT tracking, unique vehicle counting | Complete |
| 3 | Line crossing, polygon zones, trajectories, geometry tests | Next |
| 4 | Density, congestion classification, queue length, speed estimation | Planned |
| 5 | SQLite persistence, pandas analytics, `model_predictions_audit` table | Planned |
| 6 | FastAPI backend with Prometheus metrics endpoint | Planned |
| 7 | Privacy/anonymisation layer (on by default) | Planned |
| 8 | MLflow experiment tracking, champion/challenger comparison | Planned |
| 9 | Class-distribution drift detection | Planned |
| 10 | Docker, docker-compose (app + Prometheus + Grafana), GitHub Actions CI | Planned |
| 11 | ONNX export and PyTorch vs ONNX Runtime latency benchmark | Planned |
| 12 | README polish, `MODEL_CARD.md`, `training/` docs, profiling | Planned |

## Attribution

Sample clip: [`person-bicycle-car-detection.mp4`](https://github.com/intel-iot-devkit/sample-videos)
from Intel's IoT DevKit sample videos. Detection model: [Ultralytics
YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`, AGPL-3.0 — note that Ultralytics'
licence terms apply to any deployment of this project that uses their weights).
