# COMSIGNS — Scripts Usage Reference

This document lists all top-level scripts in `scripts/` and their available CLI flags.

Format: `script` — short description
Flags: list


## scripts/infer.py — Run inference on preprocessed sample
- `--checkpoint, -c` (Path, required): Path to model checkpoint (.pt)
- `--class-mapping, -m` (Path): Path to class_mapping.json
- `--input, -i` (Path, required): Input file (.pkl, .pt, .json)
- `--topk, -k` (int): Number of top predictions to show (default: 5)
- `--device, -d` (str): Device (`cpu`, `cuda`, `mps`)
- `--json` (flag): Output result as JSON
- `--verbose, -v` (flag): Verbose output

Example:
```
python scripts/infer.py --checkpoint experiments/micro_v1/checkpoints/best.pt --input samples/example.pkl --topk 5
```


## scripts/infer_video.py — Run inference from a video file
- `--video, -v` (Path, required): Path to video file
- `--model` (Path): Path to `.pt` checkpoint (default: `experiments/micro_v1/best.pt`)
- `--mapping` (Path): Class mapping JSON (default: `experiments/micro_v1/class_mapping.json`)
- `--device` (str): Device (`cpu`, `cuda`)
- `--topk` (int): Number of top predictions to show (default: 5)

Example:
```
python scripts/infer_video.py --video data/raw/.../comer_1001.mp4 --topk 3
```


## scripts/test_micro_model.py — Quick validation on micro_v1
- `--samples` (int): Number of samples to test (default: 10)
- `--model` (Path): Checkpoint path (default: `experiments/micro_v1/best.pt`)
- `--mapping` (Path): Class mapping JSON (default: `experiments/micro_v1/class_mapping.json`)
- `--device` (str): Device (default: `cpu`)


## scripts/train_v1.py / scripts/train.py / scripts/train_micro.py — Training scripts
Note: Training scripts have extensive flags. Common ones include:
- `--epochs` (int)
- `--batch-size` (int)
- `--lr` (float)
- `--device` (`auto`|`cuda`|`mps`|`cpu`)
- `--seed` (int)
- `--min-support` (int)
- `--augment` / `--no-augment` (flags)
- `--class-weighting` / `--no-class-weighting` (flags)
- `--dropout`, `--weight-decay`, `--label-smoothing`
- `--lr-scheduler` (`none`|`plateau`|`cosine`) and related scheduler params
- `--output-dir` (Path)
- `--eval` (flag)

Refer to individual scripts for full list and defaults.


## scripts/extract_samples.py — Export inference-ready samples
- Flags include dataset root, split selection, output directory, and options to extract single or multiple samples.


## scripts/extract_samples.py, scripts/analyze_dataset.py, scripts/extract_* — Data tooling
- These scripts typically provide flags for `--dataset-root`, `--split-file`, `--out-dir`, `--limit`, and `--verbose`.


## scripts/test_e2e_inference.py — End-to-end inference test
- Uses backend services. No CLI flags documented; run as-is for quick smoke tests.


## Running scripts reliably
- Preferred working directory: repository root.
- If you get import errors, run with `PYTHONPATH=src` or from repo root so modules under `src/comsigns` or `services/` resolve.


---

If you want, I can auto-generate a more detailed per-script section by parsing `parser.add_argument` entries in each script and including full default values and help text — should I proceed and replace this file with the full automated output?