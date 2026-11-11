# Drishti Mars Terrain Object Detection Suite

Ultra-lightweight, edge-optimized object detection pipeline designed to run in real-time on **Raspberry Pi 4 / 5** and drone/rover companion computers using native OpenCV DNN (ARM NEON) without heavy PyTorch or GPU dependencies.

---

## Directory Structure

```text
├── mars_terrain_nano.onnx    # Tailored ONNX model (Classes: payload_target, traffic_cone, landing_circle, drone_shadow)
├── batch_detect.py           # CLI tool: processes any image folder and outputs annotated frames, crops & CSV
├── live_final.py             # Local live testing tool for FPV receiver feed or webcam with tactical HUD
├── dataset_builder.py        # Dataset generator with realistic aerial drone augmentations
├── train_tailored_model.py   # One-click trainer to fine-tune and export tailored .onnx models
├── mars_terrain/             # Source input dataset (13 raw aerial images)
├── datasets/                 # Augmented YOLO dataset (train/val splits with bounding box labels)
├── outputs/                  # Active detection runs (annotated frames, crops, CSV logs, telemetry)
│   └── batch_detections/
│       ├── annotated/        # Bounding box & telemetry HUD overlay images
│       ├── crops/            # High-res image crops sorted by detected class
│       ├── detections.csv    # Coordinate, bounding box, area & confidence log
│       └── summary.json      # High-level metrics & FPS report
└── archive/                  # Unused legacy files, older OWLv2 scripts & intermediate runs
    ├── legacy_code/          # main.py, main_v2.py, live_owlv2.py, terrain_detector.py
    ├── legacy_assets/        # Generic COCO weights, screenshots, zip archives
    └── legacy_runs/          # Intermediate test outputs (output_001..003, test_refined, runs)
```

---

## What Each Component Does

| File / Folder | Role & Description |
| :--- | :--- |
| **`mars_terrain_nano.onnx`** | **The Active Production Model**. Tailored specifically for terrain targets (`payload_target`, `traffic_cone`, `landing_circle`, `drone_shadow`). Only ~12MB, runs at ~30 FPS on CPU via ARM NEON. |
| **`batch_detect.py`** | **Folder-to-Folder Batch Processor**. Takes any folder of images (e.g. `--input-dir mars_terrain/`), runs the ONNX model, and saves annotated images, cropped target patches, and telemetry CSV in `--output-dir`. |
| **`live_final.py`** | **FPV Live Testing CLI**. Connects to your local webcam or FPV analog video receiver (`--source 0`), overlays real-time detection boxes, and saves snapshots (`s`), toggles shadow filtering (`f`), or adjusts confidence (`c`). |
| **`dataset_builder.py`** | **Augmentation Pipeline**. Converts raw terrain images and ground truth bounding boxes into standard YOLO format with aerial rotations (90°, 180°, 270°), flips, and photometric noise. |
| **`train_tailored_model.py`** | **Model Training & Exporter**. Fine-tunes YOLO Nano on GPU/CPU and exports directly to a streamlined `.onnx` model. |
| **`archive/`** | **Storage for Older Artifacts**. Houses previous high-latency OWLv2 implementations, generic 80-class COCO weights, and previous test runs so your working tree remains clean. |

---

## Quick Start Commands

### 1. Run Batch Detection on Any Folder
```bash
python batch_detect.py --input-dir mars_terrain/ --output-dir outputs/batch_detections --conf 0.25
```

### 2. Run Live FPV / Webcam Feed Testing
```bash
# Test on USB camera / FPV receiver (/dev/video0)
python live_final.py --source 0

# Test on a recorded video file
python live_final.py --source flight_clip.mp4
```

### 3. Re-train / Fine-Tune Model on New Data
```bash
# 1. Regenerate augmented dataset
python dataset_builder.py

# 2. Train and export new ONNX model
python train_tailored_model.py
```
