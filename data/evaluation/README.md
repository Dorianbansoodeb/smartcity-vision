# Evaluation artefacts

Numbers in the README come from these files. Re-run the commands rather than
editing the JSON by hand.

## Detection quality (labelled)

`coco_val80_slice.json` is an 80-image COCO val2017 subset (seed 0) filtered to
`person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck`. It is **not** the
full 5k-image val set. Do not compare `map50` here to published COCO numbers.

```bash
python scripts/evaluate_coco.py --device cpu
```

Downloads (gitignored) land in `cache/` and `images/`. The committed slice
means the annotation zip is only fetched on the first machine that does not
already have `coco_val80_slice.json`.

`coco_val80_results.json` is the measured output of that script on this machine.

## Second clip (unlabelled, operational only)

```bash
curl -L -o data/input/cars.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4
python scripts/run_video.py --config config/car_detection.yaml
```

`cars_clip_results.json` is copied from that run. No precision / recall / mAP
— the clip has no labels.
