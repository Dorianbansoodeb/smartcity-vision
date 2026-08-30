# Fine-tuning SmartCity Vision

v1 runs the official Ultralytics `yolov8n.pt` checkpoint. Custom training is
supported structurally — drop a dataset next to this file, point
`model.weights` at the resulting `.pt`, and the rest of the pipeline does not
change.

## Dataset layout

```
data/custom/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Labels are YOLO format (`class x_center y_center width height`, all normalised).
`data.yaml` in this folder is the Ultralytics config the trainer reads.

## Train

```bash
yolo detect train data=training/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Copy the best checkpoint to `models/custom.pt` and run:

```bash
python scripts/run_video.py --weights models/custom.pt
```

## Target classes beyond COCO

The template `data.yaml` includes classes a city operations team would actually
need and that COCO does not provide:

- `emergency_vehicle`
- `damaged_vehicle`
- `stopped_vehicle`
- `accident`
- `illegally_parked_vehicle`

None of these have been trained or evaluated in this repository. Do not quote
precision, recall, or mAP for them; those numbers do not exist yet.

## Evaluation rule

When you do evaluate a custom checkpoint, record the command, the dataset
revision, and the numbers the evaluator printed. The `MODEL_CARD.md` and the
README only accept figures produced that way.

The current checkpoint was scored with:

```bash
python scripts/evaluate_coco.py --device cpu
```

That writes `data/evaluation/coco_val80_results.json`. Swap `--weights` (via
config) when you have a custom `.pt`; do not paste a number you did not just
generate.
