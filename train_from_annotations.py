"""
Automated Dataset Builder & ONNX Trainer from User Annotations.
Takes user-labeled images, applies aerial drone augmentations, trains a tailored YOLO model,
and outputs 'mars_terrain_nano.onnx' ready for OpenCV DNN and live feed deployment.
"""

import argparse
import glob
import os
import shutil
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEFAULT_CLASSES = ["object", "cone", "landing_mark"]


def augment_image_and_boxes(img, boxes, rng):
    """
    Applies realistic aerial augmentations: flips, rotations, and photometric jitter.
    boxes is list of [cls_id, cx, cy, w, h] normalized.
    """
    out_img = img.copy()
    out_boxes = [list(b) for b in boxes]
    h, w = img.shape[:2]

    # Horizontal flip
    if rng.random() > 0.5:
        out_img = cv2.flip(out_img, 1)
        for b in out_boxes:
            b[1] = 1.0 - b[1]

    # Vertical flip
    if rng.random() > 0.5:
        out_img = cv2.flip(out_img, 0)
        for b in out_boxes:
            b[2] = 1.0 - b[2]

    # 90-degree rotation
    rot_choice = rng.choice([0, 1, 2, 3])
    if rot_choice == 1:  # 90 deg clockwise
        out_img = cv2.rotate(out_img, cv2.ROTATE_90_CLOCKWISE)
        for b in out_boxes:
            cx, cy, bw, bh = b[1], b[2], b[3], b[4]
            b[1], b[2], b[3], b[4] = 1.0 - cy, cx, bh, bw
    elif rot_choice == 2:  # 180 deg
        out_img = cv2.rotate(out_img, cv2.ROTATE_180)
        for b in out_boxes:
            b[1], b[2] = 1.0 - b[1], 1.0 - b[2]
    elif rot_choice == 3:  # 270 deg
        out_img = cv2.rotate(out_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for b in out_boxes:
            cx, cy, bw, bh = b[1], b[2], b[3], b[4]
            b[1], b[2], b[3], b[4] = cy, 1.0 - cx, bh, bw

    # Photometric variation
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-18, 18)
    jittered = np.clip(out_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return jittered, out_boxes


def build_augmented_dataset(
    images_dir="mars_terrain",
    labels_dir="datasets/user_labels",
    output_dir="datasets/yolo_dataset",
    classes=None,
    num_augs_per_image=10,
):
    classes = classes or DEFAULT_CLASSES
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp"]
    img_files = []
    for pat in patterns:
        img_files.extend(glob.glob(os.path.join(images_dir, pat)))
        img_files.extend(glob.glob(os.path.join(images_dir, pat.upper())))
    img_files = sorted(list(set(img_files)))

    rng = np.random.RandomState(42)
    sample_id = 0

    print(f"\nBuilding augmented dataset from {len(img_files)} source images in '{images_dir}'...")

    for idx, p in enumerate(img_files):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        base_name = os.path.splitext(os.path.basename(p))[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")

        boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        # Validation split: 1 out of every 6 images
        split = "val" if (idx % 6 == 0 and len(boxes) > 0) else "train"

        # 1. Save original
        s_name = f"sample_{sample_id:04d}_orig"
        cv2.imwrite(os.path.join(output_dir, "images", split, f"{s_name}.jpg"), img)
        with open(os.path.join(output_dir, "labels", split, f"{s_name}.txt"), "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
        sample_id += 1

        # 2. Augmentations
        augs = 2 if split == "val" else num_augs_per_image
        for a in range(aus := augs):
            aug_img, aug_boxes = augment_image_and_boxes(img, boxes, rng)
            s_aug_name = f"sample_{sample_id:04d}_aug{a}"
            cv2.imwrite(os.path.join(output_dir, "images", split, f"{s_aug_name}.jpg"), aug_img)
            with open(os.path.join(output_dir, "labels", split, f"{s_aug_name}.txt"), "w") as f:
                for b in aug_boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            sample_id += 1

    # Write dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    abs_out = os.path.abspath(output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"Dataset generated at '{output_dir}':")
    print(f" - Train images: {len(os.listdir(os.path.join(output_dir, 'images', 'train')))}")
    print(f" - Val images:   {len(os.listdir(os.path.join(output_dir, 'images', 'val')))}")
    print(f" - Classes: {classes}")
    return yaml_path


def build_and_train_onnx(
    images_dir="mars_terrain",
    labels_dir="datasets/user_labels",
    classes=None,
    output_onnx="mars_terrain_nano.onnx",
    epochs=35,
    imgsz=480,
    batch=16,
):
    classes = classes or DEFAULT_CLASSES
    yaml_path = build_augmented_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        classes=classes,
    )

    device = 0 if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print(f"Training YOLO Nano on {device} ({torch.cuda.get_device_name(0) if device == 0 else 'CPU'})...")
    print("=" * 60)

    # Use base YOLOv8n
    base_model = "yolov8n.pt"
    if not os.path.exists(base_model):
        base_model = "archive/legacy_assets/yolov8n.pt"
        if not os.path.exists(base_model):
            base_model = "yolov8n.pt"

    model = YOLO(base_model)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/user_train",
        name="custom_model",
        exist_ok=True,
        verbose=True,
        patience=15,
        save=True,
        plots=False,
    )

    # Locate best.pt
    run_dir = "runs/detect/runs/user_train/custom_model/weights"
    if not os.path.exists(run_dir):
        run_dir = "runs/user_train/custom_model/weights"
    best_pt = os.path.join(run_dir, "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(run_dir, "last.pt")

    print("\n" + "=" * 60)
    print(f"Exporting checkpoint '{best_pt}' to ONNX...")
    print("=" * 60)
    trained_model = YOLO(best_pt)
    exported = trained_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=17,
        simplify=True,
    )

    shutil.copy(exported, output_onnx)
    print(f"\n[SUCCESS] Custom tailored ONNX model saved to: {output_onnx} ({os.path.getsize(output_onnx) / 1024 / 1024:.2f} MB)")

    # Validate with OpenCV DNN
    net = cv2.dnn.readNetFromONNX(output_onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print(f"[VERIFIED] Model loaded successfully in OpenCV DNN with classes: {classes}")
    print(f"[READY] Live feed & batch detection now automatically use '{output_onnx}'!")
    return output_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom ONNX from user annotations")
    parser.add_argument("-i", "--images-dir", default="mars_terrain", help="Folder with raw images")
    parser.add_argument("-l", "--labels-dir", default="datasets/user_labels", help="Folder with user label .txt files")
    parser.add_argument("-o", "--output-onnx", default="mars_terrain_nano.onnx", help="Output ONNX filename")
    parser.add_argument("-e", "--epochs", type=int, default=35, help="Number of epochs")
    args = parser.parse_args()

    build_and_train_onnx(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_onnx=args.output_onnx,
        epochs=args.epochs,
    )
