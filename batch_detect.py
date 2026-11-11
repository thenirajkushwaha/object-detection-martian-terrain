"""
Batch Terrain Object Detection CLI.
Runs the tailored ONNX model on any input folder of images and organizes
annotated images, cropped object patches, and telemetry CSV logs into the target folder.

Usage:
  python batch_detect.py --input-dir mars_terrain/ --output-dir outputs/batch_run_01
  python batch_detect.py -i /path/to/incoming_frames -o outputs/survey_01 --conf 0.30
"""

import argparse
import csv
import glob
import json
import os
import time
import cv2
import numpy as np


class TerrainDetectorONNX:
    """
    Ultra-efficient inference engine using OpenCV DNN.
    Executes tailored ONNX model natively with ARM NEON / x86 vectorization.
    Zero PyTorch dependency needed for inference.
    """

    def __init__(
        self,
        onnx_model_path="mars_terrain_nano.onnx",
        input_size=(480, 480),
        conf_threshold=0.25,
        nms_threshold=0.45,
        class_names=None,
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or [
            "payload_target",
            "traffic_cone",
            "landing_circle",
            "drone_shadow",
        ]

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model '{onnx_model_path}' not found! Run train_tailored_model.py first.")

        print(f"Loading ONNX Model via OpenCV DNN: {onnx_model_path}")
        self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, img_bgr):
        t0 = time.time()
        h, w = img_bgr.shape[:2]
        in_w, in_h = self.input_size

        # Letterbox scaling (preserves aspect ratio)
        scale = min(in_w / w, in_h / h)
        nw, nh = int(w * scale), int(h * scale)
        pad_x, pad_y = (in_w - nw) // 2, (in_h - nh) // 2

        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (in_w, in_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        # Handle output dimensions: (1, 4 + num_classes, N) -> (N, 4 + num_classes)
        if preds.ndim == 3:
            preds = preds[0]
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T

        boxes, confidences, class_ids = [], [], []
        num_classes = len(self.class_names)

        for row in preds:
            class_scores = row[4: 4 + num_classes]
            max_score = float(np.max(class_scores))
            if max_score >= self.conf_threshold:
                cls_id = int(np.argmax(class_scores))
                cx, cy, bw, bh = row[0:4]

                # Map back to original image coordinates
                orig_cx = (cx - pad_x) / scale
                orig_cy = (cy - pad_y) / scale
                orig_w = bw / scale
                orig_h = bh / scale

                x1 = int(orig_cx - orig_w / 2.0)
                y1 = int(orig_cy - orig_h / 2.0)
                w_box = int(orig_w)
                h_box = int(orig_h)

                boxes.append([x1, y1, w_box, h_box])
                confidences.append(max_score)
                class_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections = []
        if len(indices) > 0:
            for n, idx in enumerate(indices.flatten(), 1):
                x, y, bw, bh = boxes[idx]
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                bw = max(2, min(w - x, bw))
                bh = max(2, min(h - y, bh))
                cls_idx = class_ids[idx]
                label = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"

                detections.append({
                    "id": n,
                    "label": label,
                    "score": float(confidences[idx]),
                    "box": [x, y, bw, bh],
                    "center": [int(x + bw / 2), int(y + bh / 2)],
                    "area": bw * bh,
                })

        latency_ms = (time.time() - t0) * 1000.0
        return detections, latency_ms


def draw_hud(img, detections, latency_ms=0.0, filename=""):
    """Draws sleek tactical HUD bounding boxes with telemetry header."""
    out = img.copy()
    h, w = out.shape[:2]

    color_palette = {
        "payload_target": (255, 255, 0),    # Cyan/Aqua in BGR
        "traffic_cone": (0, 140, 255),      # Vivid Orange
        "landing_circle": (200, 255, 200),  # Mint/Chalk
        "drone_shadow": (120, 120, 120),    # Stealth Grey
        "default": (0, 255, 128),
    }

    for d in detections:
        x, y, bw, bh = d["box"]
        label = d["label"]
        score = d["score"]
        col = color_palette.get(label, color_palette["default"])

        # Main bounding box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)

        # Tactical corner accents
        line_len = min(12, bw // 4, bh // 4)
        if line_len > 2:
            cv2.line(out, (x, y), (x + line_len, y), col, 3)
            cv2.line(out, (x, y), (x, y + line_len), col, 3)
            cv2.line(out, (x + bw, y), (x + bw - line_len, y), col, 3)
            cv2.line(out, (x + bw, y), (x + bw, y + line_len), col, 3)
            cv2.line(out, (x, y + bh), (x + line_len, y + bh), col, 3)
            cv2.line(out, (x, y + bh), (x, y + bh - line_len), col, 3)
            cv2.line(out, (x + bw, y + bh), (x + bw - line_len, y + bh), col, 3)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - line_len), col, 3)

        # Label banner
        tag = f"#{d['id']} {label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 5, y)
        cv2.rectangle(out, (x, ty - th - 5), (x + tw + 6, ty + 2), col, -1)
        cv2.putText(out, tag, (x + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Top telemetry bar
    bar_h = 32
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    fps = 1000.0 / max(latency_ms, 0.001)
    status = f"[{filename}] | Targets: {len(detections)} | {latency_ms:.1f}ms ({fps:.1f} FPS)"
    cv2.putText(out, status, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def process_folder(
    input_dir,
    output_dir,
    model_path="mars_terrain_nano.onnx",
    conf_threshold=0.25,
    nms_threshold=0.45,
    save_crops=True,
    filter_shadows=False,
):
    """
    Scans input_dir for images, runs detection with tailored ONNX model,
    and populates output_dir with annotated images, cropped targets, and telemetry CSV.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    # Find supported images
    patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pat)))
        image_paths.extend(glob.glob(os.path.join(input_dir, pat.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No images found in '{input_dir}'.")
        return

    # Setup output directory structure
    annotated_dir = os.path.join(output_dir, "annotated")
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(annotated_dir, exist_ok=True)
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "detections.csv")
    json_path = os.path.join(output_dir, "summary.json")

    # Initialize ONNX Detector
    detector = TerrainDetectorONNX(
        onnx_model_path=model_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    print("\n" + "=" * 70)
    print(f"Starting Batch Detection on {len(image_paths)} images...")
    print(f" Input:  {os.path.abspath(input_dir)}")
    print(f" Output: {os.path.abspath(output_dir)}")
    print("=" * 70)

    total_detections = 0
    class_counts = {}
    latencies = []
    csv_rows = []

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[{idx}/{len(image_paths)}] Failed to read: {filename}")
            continue

        detections, dt = detector.detect(img)
        latencies.append(dt)

        if filter_shadows:
            detections = [d for d in detections if d["label"] != "drone_shadow"]

        # Save annotated image
        annotated_img = draw_hud(img, detections, dt, filename)
        save_name = os.path.splitext(filename)[0] + "_det.jpg"
        cv2.imwrite(os.path.join(annotated_dir, save_name), annotated_img)

        # Save crops
        if save_crops:
            for d in detections:
                x, y, bw, bh = d["box"]
                crop = img[y:y + bh, x:x + bw]
                c_dir = os.path.join(crops_dir, d["label"])
                os.makedirs(c_dir, exist_ok=True)
                crop_name = f"{os.path.splitext(filename)[0]}_obj{d['id']}_{d['score']:.2f}.jpg"
                cv2.imwrite(os.path.join(c_dir, crop_name), crop)

        # Record CSV rows
        for d in detections:
            x, y, bw, bh = d["box"]
            cx, cy = d["center"]
            csv_rows.append([
                filename,
                d["id"],
                d["label"],
                round(d["score"], 4),
                x, y, bw, bh,
                cx, cy,
                d["area"],
                round(dt, 2),
            ])
            class_counts[d["label"]] = class_counts.get(d["label"], 0) + 1

        total_detections += len(detections)
        print(f"[{idx:02d}/{len(image_paths):02d}] {filename:<44} -> {len(detections)} targets ({dt:5.1f}ms)")

    # Write CSV telemetry
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "id", "label", "confidence", "x", "y", "width", "height",
            "center_x", "center_y", "area_px", "latency_ms"
        ])
        writer.writerows(csv_rows)

    # Write JSON summary
    summary = {
        "input_directory": os.path.abspath(input_dir),
        "output_directory": os.path.abspath(output_dir),
        "total_images": len(image_paths),
        "total_detections": total_detections,
        "class_breakdown": class_counts,
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "mean_fps": round(1000.0 / float(np.mean(latencies)), 1) if latencies else 0.0,
        "model_path": model_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Batch Detection Complete!")
    print(f" - Total Images Processed: {len(image_paths)}")
    print(f" - Total Objects Found:    {total_detections}")
    print(f" - Class Breakdown:        {class_counts}")
    print(f" - Average Inference:      {summary['mean_latency_ms']} ms ({summary['mean_fps']} FPS)")
    print(f" - Annotated Images:       {annotated_dir}")
    print(f" - Object Crops:           {crops_dir}")
    print(f" - Telemetry Log CSV:      {csv_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Batch Terrain Object Detection CLI")
    parser.add_argument("-i", "--input-dir", default="mars_terrain", help="Folder containing images to process")
    parser.add_argument("-o", "--output-dir", default="outputs/batch_results", help="Destination folder for detections")
    parser.add_argument("-m", "--model", default="mars_terrain_nano.onnx", help="Path to tailored ONNX model")
    parser.add_argument("-c", "--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("-n", "--nms", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--no-crops", action="store_true", help="Do not save cropped object patches")
    parser.add_argument("--filter-shadows", action="store_true", help="Suppress drone shadow detections")
    args = parser.parse_args()

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        save_crops=not args.no_crops,
        filter_shadows=args.filter_shadows,
    )


if __name__ == "__main__":
    main()
