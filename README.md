# SmartCity Vision

[![CI](https://github.com/Dorianbansoodeb/smartcity-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/Dorianbansoodeb/smartcity-vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg)](https://www.python.org/downloads/)

Turns traffic-camera video into structured traffic analytics with YOLOv8 and OpenCV,
then treats the result like a regulated production system: every run is persisted
with a model hash and a config snapshot, frames are anonymised before they hit
disk, class-mix drift is scored, and inference latency is scraped in Prometheus
text format.

**Status: Phases 1–12 complete.** Every number below was produced by running the
code on this machine. Nothing is estimated.

**Live demo:** [dorian-smartcity-vision.fly.dev](https://dorian-smartcity-vision.fly.dev)
shows a pre-rendered annotated run immediately; Analyze re-runs the model on the
server. GitHub Pages is a static replay only.

![Tracked car with a persistent ID](docs/images/phase2_tracked_frame.png)

## Quickstart

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

curl -L -o data/input/traffic.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4

python scripts/run_video.py --input data/input/traffic.mp4
```

The first run downloads `yolov8n.pt` (6.2 MB) into `models/`. Outputs land in
`data/output/`:

| File | Contents |
| --- | --- |
| `annotated_video.mp4` | Boxes, track IDs, trajectories, lines, zones, HUD |
| `run_summary.json` | Measured stats + the full config snapshot |
| `smartcity.db` | Detections, events, metrics, `model_predictions_audit` |
| `events.csv` / `traffic_metrics.csv` / `summary.json` | Pandas export of that run |

```bash
pytest                    # 162 tests, ~2 s, no GPU required
ruff check . && ruff format --check .
mypy                      # package-wide, disallow_untyped_defs
python -m uvicorn smartcity_vision.api.app:app --reload   # API + live demo UI
python scripts/benchmark_onnx.py                          # PyTorch vs ONNX, real numbers
python scripts/evaluate_coco.py                           # precision / recall / mAP@50
```

## What the sample clip actually produced

Clip: 647 frames, 768×432, 12 fps. Weights: `yolov8n.pt`. Tracker: ByteTrack.

| Quantity | Measured value |
| --- | --- |
| Detections | 344 |
| Unique objects | 10 (2 bicycle, 3 car, 5 person) from 11 tracks; 1 discarded as too short-lived |
| Line crossings | 3, all `lot_midline`, all `A->B` |
| Zone events | 13 enter, 13 exit (`parking_lot`) |
| Peak congestion | LOW |
| Mean estimated speed | 81.5 px/s (uncalibrated — not km/h) |
| Persist | run `d7ba019b5f8d4dcba13fc1d5ffa3bb8f`, 344 detection rows, 647 metric rows |

Unique counts are *reproducible*, not *validated*. Continuity of track IDs on this
clip was 0.98 mean (max span 68 frames). See Phase 2 notes below.

## Detection quality (COCO val slice)

The parking-lot clip has no labels, so it cannot produce precision, recall, or
mAP. Those metrics were measured with this repo's matcher — not `model.val()` —
on an **80-image COCO val2017 slice** (seed 0) containing the road-user classes
we actually filter to. Protocol, command, and JSON: [`data/evaluation/`](data/evaluation/README.md).

This is not the 5k-image COCO val set. Do not compare `map50` here to published
YOLOv8n COCO scores.

Weights: `yolov8n.pt`. Device: CPU. Confidence 0.25, NMS IoU 0.45, `imgsz=640`.
Match IoU for scoring: 0.50. Crowd boxes are ignore regions.

| | mAP@50 | Precision | Recall |
| --- | --- | --- | --- |
| **Aggregate** | **0.4419** | **0.7653** | **0.5296** |

Per class (operating-point P/R, plus 101-point AP@50):

| Class | GT | AP@50 | P | R |
| --- | --- | --- | --- | --- |
| person | 303 | 0.565 | 0.790 | 0.607 |
| bicycle | 25 | 0.260 | 0.643 | 0.360 |
| car | 155 | 0.414 | 0.711 | 0.445 |
| motorcycle | 42 | 0.517 | 0.815 | 0.524 |
| bus | 20 | 0.565 | 0.800 | 0.600 |
| truck | 46 | 0.330 | 0.739 | 0.370 |

What this actually says: at the deployed threshold the model is **precise and
incomplete**. It misses about half the labelled objects (recall 0.53), and the
small classes (bicycle, truck) are worse. Person and bus are the strongest.
That is a reason to tune the threshold or fine-tune, not a reason to quote a
paper's full-val mAP as if it applied to this pipeline.

```bash
python scripts/evaluate_coco.py --device cpu
```

## Second clip (different scene)

Same weights and tracker, different video: Intel IoT `car-detection.mp4`
(377 frames, 768×432, 12.5 fps — a road, not a parking lot). Geometry from the
first clip does not apply, so this run used [`config/car_detection.yaml`](config/car_detection.yaml)
with empty lines/zones. Unlabelled: operational metrics only.

| Quantity | Parking-lot clip | Road clip |
| --- | --- | --- |
| Frames | 647 @ 12 fps | 377 @ 12.5 fps |
| Detections | 344 | 68 |
| Unique objects | 10 (2 bicycle, 3 car, 5 person) | 6 (5 car, 1 bus) |
| Class mix | people + bikes + cars | car-heavy; 1 person box in the whole clip |
| Peak congestion | LOW | LOW (last rolling density 0.0 — vehicles had left) |
| Throughput | 47.15 fps ByteTrack / MPS | 46.28 fps ByteTrack / MPS |
| Persist | `d7ba019b…` | `675d1907f84e4e0c83608e5dadaadce6` |

The pipeline ran without retuning the detector. The counts moved because the
scene moved. That is the point of a second clip.

```bash
curl -L -o data/input/cars.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4
python scripts/run_video.py --config config/car_detection.yaml
```

## Measured performance

Absolute latency on a fanless laptop drifts with thermal state. Compare **within**
a batch; ranges sit next to medians.

Hardware: Apple M3, macOS 26.5.1, Python 3.13.14, torch 2.13.0, ultralytics 8.4.135,
OpenCV 5.0.0.

**Device comparison** (detection only, 3 repeats):

| Device | Avg inference | p95 | Throughput |
| --- | --- | --- | --- |
| MPS | 16.93 ms [16.57–18.04] | 25.22 ms | 55.34 fps |
| CPU | 22.32 ms [22.04–22.59] | 25.65 ms | 43.74 fps |

**Tracker comparison** (MPS, interleaved):

| Configuration | Avg inference | Throughput | Unique |
| --- | --- | --- | --- |
| Detection only | 18.85 ms | 49.45 fps | n/a |
| ByteTrack | 19.65 ms | 47.15 fps | 10 |
| BoT-SORT | 20.96 ms | 44.70 fps | 10 |

Tracking costs ~1–2 ms/frame here. Both trackers produced the same unique counts.

**ONNX vs PyTorch** (`scripts/benchmark_onnx.py`, 30 frames after 5 warmup, CPU,
`imgsz=640`):

| Runtime | Avg | p95 | min | max |
| --- | --- | --- | --- | --- |
| PyTorch | 29.72 ms | 55.37 ms | 22.59 ms | 126.33 ms |
| ONNX Runtime 1.29.0 | 39.74 ms | 43.93 ms | 36.65 ms | 51.89 ms |

ONNX was **slower on the mean** (0.75×) and **tighter on the tail**. That is the
honest answer to "how would you cut latency": export, measure, and notice that
mean and p95 can move in opposite directions. A production call would then
profile the Ultralytics ORT wrapper vs a slim custom runtime, not assume ONNX
is faster.

Latency is measured after warmup. The first MPS forward pass took 742–1425 ms
across four runs (~45–85× steady state).

## Architecture

```
src/smartcity_vision/
├── detection/     YoloDetector, YoloTracker (ByteTrack / BoT-SORT)
├── video/         source abstraction + VideoProcessor
├── analytics/     count, lines, zones, trajectories, density, queue, speed
├── privacy/       face/plate redaction (on by default, before disk write)
├── monitoring/    Prometheus instruments + class-mix drift
├── database/      SQLite + model_predictions_audit
├── reports/       pandas summary + CSV/JSON export
├── api/           FastAPI, Pydantic, live demo UI, /metrics/prometheus
├── evaluation/    IoU matching, AP@50, COCO-slice runner
├── visualization/ overlays from config, no hardcoded coordinates
└── experiments.py MLflow logging + champion/challenger report
```

Four boundaries keep later changes local:

1. A `Frame` does not know if it came from a file, a webcam, or RTSP.
2. Ultralytics results become `Detection` objects at the edge.
3. `YoloTracker` subclasses `YoloDetector` and overrides one method.
4. `VideoProcessor` is the only place that knows the order of operations.

## Privacy

`privacy.enabled` defaults to **true**. Annotated video is redacted before it is
written. OpenCV 5 wheels on this machine do not ship Haar cascades, so the
anonymiser falls back to geometry: the top of each person box and the bottom of
each vehicle box. That is a real pixel change (the tests assert it) and a
deliberately conservative stand-in, not a production PII detector. See
`MODEL_CARD.md`.

Public camera footage of people is exactly the data GDPR/PIPEDA-style rules
exist for. The control is on by default so a forgotten flag cannot leak a face
into `data/output/`.

## Monitoring and drift

`GET /metrics/prometheus` exposes inference latency, request count, per-class
detections, and completed-run counts.

`DriftDetector` compares the class mix in a rolling window against a stored
baseline using total variation distance. An injected all-person window against
an 80/20 car/person baseline scores TV = 0.80 and is flagged. In production
this is the check Prometheus Alertmanager would page on; this repo does not
build the pager.

## API

```bash
python -m uvicorn smartcity_vision.api.app:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/metrics` | Latest-run summary |
| GET | `/metrics/prometheus` | Prometheus text |
| GET | `/vehicles/counts` | Unique counts |
| GET | `/traffic/current` | Last metric row |
| GET | `/traffic/history` | Full metric series |
| GET | `/crossings` | Line crossings |
| GET | `/zones/events` | Zone enter/exit |
| POST | `/analyze/video` | Run the pipeline |
| GET | `/reports/summary` | Pandas summary |
| GET | `/reports/export` | Rewrite CSV/JSON |

Interactive docs: `/docs`.

## Configuration

Everything lives in [`config/default.yaml`](config/default.yaml) and is validated
by Pydantic. Unknown keys are rejected. Current sections: `model`, `tracking`,
`analytics` (counting, lines, zones, trajectories, density, queue, speed),
`video`, `output`, `visualization`, `privacy`, `monitoring`, `logging`.

Lines and zones are pixel coordinates for the 768×432 sample clip. Retune per
camera — nothing in the code hardcodes them.

## Testing

```bash
pytest          # 162 tests
ruff check .
ruff format --check .
mypy
```

Tests assert behaviour: a 50-frame track counts once; walking past the end of a
line is not a crossing; a vanished-inside-zone track is closed as an exit;
history length is capped; congestion thresholds are inclusive of the lower
bound; injected distribution shift is flagged; redaction changes interior
pixels and leaves the exterior alone; `/crossings` is 404 when no run exists
rather than inventing an empty success; a perfect box match is AP@50 = 1; a
low-IoU box is a false positive; a crowd box does not create a false negative.

## Docker and CI

```bash
docker compose up --build
```

brings up the API (`:8000`), Prometheus (`:9090`), and Grafana (`:3000`).
Grafana is provisioned against Prometheus; the dashboard is a starting point,
not a claim that it has been used in an incident.

GitHub Actions (`.github/workflows/ci.yml`) runs pytest, ruff, mypy, and a Docker
build on every push/PR.

The public live demo is `Dockerfile.demo` on Fly.io (`fly.toml`, region `yyz`).
It serves `/`, runs one analysis at a time, and transcodes the OpenCV writer
output to H.264 so the browser can play it. The machine auto-stops when idle.

```bash
fly deploy
```

## How this scales past one machine

This repository is a single process on one box. That is enough to measure the
model; it is not how you serve a city. The first split is **ingest vs infer**.
Each camera publishes frames (or keyframe-sized chunks) onto a bus — Kafka or
Pub/Sub — keyed by camera id so one bad stream cannot stall the others. A
consumer group of GPU workers pulls batches, runs YOLO, and writes detections
to a second topic. Analytics (counting, crossings, drift) subscribe to
detections, not pixels, so you can replay a camera without re-decoding video.
The FastAPI process in this repo becomes a query path over that store, not the
inference loop.

Autoscaling belongs on the **Prometheus inference histogram already exposed at
`/metrics/prometheus`**, not on CPU. Horizontal pod autoscaling on p95 latency
(and on consumer-group lag) adds GPU workers when a district comes online at
rush hour and sheds them at night. Batch size is the other lever: a T4 will
waste most of its memory on `batch=1`; the same card at `batch=8–16` is the
difference between tens and hundreds of frames per second for YOLOv8n.

Order-of-magnitude cost, not a quote. 1,000 cameras at 12 fps is 12,000
frames/s. This laptop did ~47 fps end-to-end on MPS for one stream; a batched
T4 for YOLOv8n is commonly a few hundred fps. Call it 200 fps/worker and you
need on the order of 60 GPUs. At a rough \$0.40/hr for a cloud T4 that is
about \$17k/month for compute, **~$17/camera/month**, before ingest, storage,
and egress. You would not run every camera at 12 fps × 640 px all day —
motion-triggered skip, a cheaper detector for quiet suburban feeds, and a
heavier one for intersections cut that number faster than buying more cards.
None of that is deployed here. The live demo is one Fly machine that stops
when idle.

### Deploying this to Cloud Run / GKE (config only)

A Cloud Run service would wrap the same `uvicorn` image, mount a volume (or
GCS fuse) for `data/` and `models/`, and point Prometheus at
`/metrics/prometheus`. GKE would add an HPA on that latency histogram and a
dedicated node pool if you move inference to a GPU. The Fly live demo is the
deployment that has actually been stood up from this repository.

## Custom training

See [`training/README.md`](training/README.md) and [`MODEL_CARD.md`](MODEL_CARD.md).
The future class list (`emergency_vehicle`, `accident`, …) is a template.
Those classes have not been trained.

## Known limitations

- Parking-lot and road clips are unlabelled. Precision / recall / mAP@50 were
  measured on an 80-image COCO val slice, not on those videos, and not on the
  full COCO val set.
- Peak congestion LOW is correct for both measured clips.
- Speed and queue length are pixel-space unless you set `metres_per_pixel`.
- Haar cascades are missing from the OpenCV 5 wheel used here; fallback regions
  are used instead.
- Identity is not guaranteed across full occlusion.
- ONNX was not faster on the mean in the measured CPU run.
- Single-machine numbers; treat cross-batch comparisons as meaningless.

## Licence

Application code is **MIT** ([`LICENSE`](LICENSE)). The default detector
weights, `yolov8n.pt`, are **Ultralytics YOLOv8 under AGPL-3.0**. That licence
infects a networked service that loads those weights: if you ship this as a
SaaS with the official checkpoint, AGPL obligations apply to the service, not
just the `ultralytics` package.

A commercial deployment would do one of: buy an Ultralytics licence, or swap
the detector for a permissively licensed one (train a custom head and serve it
with a BSD/Apache runtime; do not load `yolov8n.pt`). This repo's MIT code is
written against a `Detection` object so that swap is a detector-module change,
not a rewrite. I have not performed that swap here.

## Attribution

Sample clips: [Intel IoT DevKit sample videos](https://github.com/intel-iot-devkit/sample-videos)
(`person-bicycle-car-detection.mp4`, `car-detection.mp4`).
Evaluation labels: [COCO 2017 val](https://cocodataset.org/#home).
Detection model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
(`yolov8n.pt`, AGPL-3.0).
