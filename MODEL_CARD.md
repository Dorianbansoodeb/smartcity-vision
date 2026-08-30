# Model card — SmartCity Vision

## Intended use

Analyse traffic-camera or road video to detect, track, and count road users
(`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`), measure line
crossings and zone occupancy, and produce structured analytics for a human
operator or a downstream API.

This is a **portfolio / research prototype**, not a safety system. It must not
be used to issue tickets, trigger emergency response, or make decisions about
named individuals.

## Out of scope

- Identifying people (face recognition, re-identification, gait)
- Reading licence-plate characters
- Legal evidence or automated enforcement
- Night-time, weather-degraded, or heavily occluded scenes (unvalidated)
- Any jurisdiction's official traffic counts

## Model

| Item | Value |
| --- | --- |
| Architecture | YOLOv8 nano (`yolov8n`) |
| Source | Ultralytics official pretrained COCO weights |
| File | `models/yolov8n.pt` (downloaded on first run) |
| Licence | AGPL-3.0 (Ultralytics terms apply to any deployment that uses these weights) |
| Task | Object detection; tracking via ByteTrack or BoT-SORT |

A SHA-256 of the weights file is stored on every run in
`model_predictions_audit.model_hash` so a result can be traced to the exact
bytes that produced it.

## Data

v1 uses the public Intel IoT DevKit clip
`person-bicycle-car-detection.mp4`. It has **no ground-truth labels** in this
repository, so precision, recall, mAP@50, and mAP@50-95 are **not reported**.
Those metrics have not been computed.

## Known limitations

- Performance is expected to degrade in low light, rain, snow, glare, unusual
  camera angles, and heavy occlusion. **This has not been measured.**
- The pretrained COCO classes do not include emergency vehicles, accidents, or
  illegally parked vehicles.
- Class flicker (car ↔ bus, person ↔ bicycle) is real; unique counting uses
  majority vote to stop it inflating totals, which is a mitigation, not a fix.
- Haar-cascade anonymisation misses profile faces, distant plates, and
  anything the cascade was not trained on. It is a compliance control, not a
  guarantee of anonymisation.
- Identity is not guaranteed across full occlusion; a reappearing vehicle may
  receive a new track ID and be counted twice.
- Speed and queue length are pixel-space approximations unless a calibration
  scale is supplied. Uncalibrated figures must not be quoted in km/h or metres.

## What a production deployment would need first

1. A labelled evaluation set covering day, night, rain, and at least two camera
   geometries, with reported precision / recall / mAP that come from that set.
2. Bias and fairness checks across lighting, weather, and camera placement —
   pedestrian recall in particular.
3. A human-review path for edge cases (stopped vehicle vs parked, emergency
   vehicle vs truck).
4. A data-retention policy for stored video and for the SQLite audit trail,
   including deletion and access control.
5. Stronger PII redaction than Haar cascades (a dedicated face/plate model,
   plus a fail-closed option that refuses to persist a frame when redaction
   confidence is low).
6. Continuous drift monitoring wired to an on-call path (Prometheus
   Alertmanager). The `DriftDetector` in this repo is the check; it is not the
   page.
7. A champion/challenger gate that requires the challenger to beat the champion
   on a frozen evaluation set before promotion — latency alone is not enough.

## Contact

Dorian Bansoodeb — see the repository README.
