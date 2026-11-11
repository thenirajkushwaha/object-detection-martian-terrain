"""
Dataset Builder & Auto-Augmenter for Mars Terrain Object Detection.
Builds a curated YOLO-format dataset with train/val splits and realistic aerial augmentations.

Classes:
  0: payload_target  (cyan / blue marker payload)
  1: traffic_cone    (orange marker cone)
  2: landing_circle  (chalk / white landing ring)
  3: drone_shadow    (quadcopter shadow)
"""

import os
import glob
import json
import shutil
import cv2
import numpy as np

CLASSES = ["payload_target", "traffic_cone", "landing_circle", "drone_shadow"]

# Ground truth annotations for the 13 raw mars_terrain images
# Coordinates are [class_id, x_min, y_min, x_max, y_max] in pixels (640x480)
RAW_ANNOTATIONS = {
    "WhatsApp Image 2026-10-21 at 19.13.39 (1).jpeg": [
        [1, 274, 342, 310, 385],   # traffic cone
        [2, 218, 305, 415, 460],   # chalk landing circle around cone
        [2, 435, 390, 515, 455],   # secondary chalk circle
    ],
    "WhatsApp Image 2026-10-21 at 19.13.39 (2).jpeg": [
        [0, 408, 164, 434, 184],   # payload_target
        [3, 375, 100, 425, 145],   # drone_shadow
    ],
    "WhatsApp Image 2026-10-21 at 19.13.39.jpeg": [
        [0, 268, 196, 310, 228],   # payload_target
        [3, 370, 94, 432, 155],    # drone_shadow
    ],
    "WhatsApp Image 2026-10-21 at 19.13.40 (1).jpeg": [
        [0, 490, 150, 520, 170],   # payload_target
        [3, 395, 102, 465, 168],   # drone_shadow
        [2, 345, 95, 465, 215],    # chalk circle
    ],
    "WhatsApp Image 2026-10-21 at 19.13.40 (2).jpeg": [
        # Horizon/barrier scene
    ],
    "WhatsApp Image 2026-10-21 at 19.13.40.jpeg": [
        [0, 470, 310, 520, 365],   # payload_target
        [3, 335, 260, 395, 310],   # drone_shadow
        [2, 280, 230, 420, 350],   # chalk circle
    ],
    "WhatsApp Image 2026-10-21 at 19.13.41 (1).jpeg": [
        [0, 335, 192, 378, 240],   # payload_target
        [3, 345, 255, 395, 300],   # drone_shadow
        [2, 290, 240, 415, 345],   # chalk circle
    ],
    "WhatsApp Image 2026-10-21 at 19.13.41.jpeg": [
        [0, 388, 202, 414, 220],   # payload_target
        [3, 114, 332, 162, 376],   # drone_shadow
    ],
    "WhatsApp Image 2026-10-21 at 19.13.42 (1).jpeg": [
        [0, 338, 194, 370, 224],   # payload_target
        [3, 340, 245, 385, 295],   # drone_shadow
        [2, 285, 220, 420, 350],   # chalk ring
    ],
    "WhatsApp Image 2026-10-21 at 19.13.42 (2).jpeg": [
        [0, 270, 220, 330, 284],   # payload_target
        [3, 360, 318, 400, 356],   # drone_shadow
    ],
    "WhatsApp Image 2026-10-21 at 19.13.42.jpeg": [
        [1, 510, 172, 525, 188],   # distant cone
        [3, 375, 330, 440, 385],   # drone_shadow
    ],
    "WhatsApp Image 2026-10-21 at 19.13.43 (1).jpeg": [
        [0, 250, 245, 282, 265],   # payload_target
        [3, 400, 108, 462, 162],   # drone_shadow
        [2, 190, 140, 395, 330],   # chalk ring
    ],
    "WhatsApp Image 2026-10-21 at 19.13.43.jpeg": [
        [0, 390, 158, 425, 180],   # payload_target
        [3, 372, 108, 430, 165],   # drone_shadow
    ],
}


def box_to_yolo(box, img_w=640, img_h=480):
    cls_id, x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / float(img_w)
    h = (y2 - y1) / float(img_h)
    return cls_id, cx, cy, w, h


def transform_box(box, transform_type, img_w=640, img_h=480):
    cls_id, x1, y1, x2, y2 = box
    if transform_type == "hflip":
        return [cls_id, img_w - x2, y1, img_w - x1, y2]
    elif transform_type == "vflip":
        return [cls_id, x1, img_h - y2, x2, img_h - y1]
    elif transform_type == "rot90":
        # 90 deg clockwise: new_x = img_h - y, new_y = x
        return [cls_id, img_h - y2, x1, img_h - y1, x2]
    elif transform_type == "rot180":
        return [cls_id, img_w - x2, img_h - y2, img_w - x1, img_h - y1]
    elif transform_type == "rot270":
        return [cls_id, y1, img_w - x2, y2, img_w - x1]
    return box


def apply_photometric_jitter(img, rng):
    """Simulates harsh sun, shadows, and camera color temperature variations."""
    out = img.astype(np.float32)
    # Brightness scale
    alpha = rng.uniform(0.75, 1.25)
    # Contrast offset
    beta = rng.uniform(-20, 20)
    out = np.clip(out * alpha + beta, 0, 255).astype(np.uint8)

    # Mild color channel jitter
    if rng.random() > 0.5:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-6, 6)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def build_dataset(base_dir="mars_terrain", output_dir="datasets/mars_terrain"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    rng = np.random.RandomState(42)
    img_files = sorted(glob.glob(os.path.join(base_dir, "*.jpeg")))

    print(f"Building augmented dataset from {len(img_files)} source images...")
    sample_count = 0

    # Pick 2 images for validation, rest for training
    val_names = {
        "WhatsApp Image 2026-10-21 at 19.13.41.jpeg",
        "WhatsApp Image 2026-10-21 at 19.13.39 (1).jpeg"
    }

    for p in img_files:
        name = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = RAW_ANNOTATIONS.get(name, [])

        is_val = (name in val_names)
        split = "val" if is_val else "train"

        # 1. Save original
        base_id = f"img_{sample_count:03d}"
        cv2.imwrite(os.path.join(output_dir, "images", split, f"{base_id}_orig.jpg"), img)
        with open(os.path.join(output_dir, "labels", split, f"{base_id}_orig.txt"), "w") as f:
            for b in boxes:
                c, cx, cy, bw, bh = box_to_yolo(b, w, h)
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        sample_count += 1

        # 2. Augment for training (and light augmentation for validation)
        num_aug = 3 if is_val else 12
        for a_idx in range(num_aug):
            aug_img = img.copy()
            aug_boxes = [list(b) for b in boxes]

            # Horizontal flip
            if rng.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)
                aug_boxes = [transform_box(b, "hflip", w, h) for b in aug_boxes]

            # Vertical flip
            if rng.random() > 0.5:
                aug_img = cv2.flip(aug_img, 0)
                aug_boxes = [transform_box(b, "vflip", w, h) for b in aug_boxes]

            # Photometric noise
            aug_img = apply_photometric_jitter(aug_img, rng)

            aug_id = f"img_{sample_count:03d}_aug{a_idx}"
            cv2.imwrite(os.path.join(output_dir, "images", split, f"{aug_id}.jpg"), aug_img)
            with open(os.path.join(output_dir, "labels", split, f"{aug_id}.txt"), "w") as f:
                for b in aug_boxes:
                    c, cx, cy, bw, bh = box_to_yolo(b, w, h)
                    f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            sample_count += 1

    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    abs_out = os.path.abspath(output_dir)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    print(f"Dataset generated at {output_dir}:")
    print(f" - Train images: {len(os.listdir(os.path.join(output_dir, 'images', 'train')))}")
    print(f" - Val images:   {len(os.listdir(os.path.join(output_dir, 'images', 'val')))}")
    print(f" - Classes: {CLASSES}")
    return yaml_path


if __name__ == "__main__":
    build_dataset()
