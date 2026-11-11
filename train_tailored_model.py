"""
Trainer & ONNX Exporter for Tailored Mars Terrain YOLO Model.
Fine-tunes YOLOv8n on the augmented mars terrain dataset and exports to an optimized ONNX model
compatible with Raspberry Pi CPU (OpenCV DNN / ARM NEON) and workstation GPUs.
"""

import os
import shutil
import time
import cv2
import torch
from ultralytics import YOLO


def train_and_export(
    dataset_yaml="datasets/mars_terrain/dataset.yaml",
    base_model="yolov8n.pt",
    output_onnx="mars_terrain_nano.onnx",
    epochs=35,
    imgsz=480,
    batch=16,
):
    print("=" * 60)
    print("Step 1: Initializing YOLO Nano for Mars Terrain Training...")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training Device: {device} ({torch.cuda.get_device_name(0) if device == 0 else 'CPU'})")

    model = YOLO(base_model)

    print(f"Fine-tuning for {epochs} epochs on {dataset_yaml} (imgsz={imgsz}, batch={batch})...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/mars_train",
        name="tailored_model",
        exist_ok=True,
        verbose=True,
        patience=15,
        save=True,
        plots=False,
    )

    best_pt = "runs/mars_train/tailored_model/weights/best.pt"
    if not os.path.exists(best_pt):
        best_pt = "runs/mars_train/tailored_model/weights/last.pt"
    print(f"Best checkpoint saved at: {best_pt}")

    print("\n" + "=" * 60)
    print("Step 2: Exporting to Edge-Optimized ONNX format...")
    print("=" * 60)
    best_model = YOLO(best_pt)
    exported_path = best_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=17,
        simplify=True,
    )
    print(f"ONNX Exported to: {exported_path}")

    # Copy to project root
    shutil.copy(exported_path, output_onnx)
    print(f"Tailored ONNX model saved as project asset: {output_onnx} ({os.path.getsize(output_onnx) / 1024 / 1024:.2f} MB)")

    print("\n" + "=" * 60)
    print("Step 3: Validating Model with OpenCV DNN (RPi Engine)...")
    print("=" * 60)
    net = cv2.dnn.readNetFromONNX(output_onnx)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    dummy_blob = cv2.dnn.blobFromImage(
        np_zeros := (cv2.imread("mars_terrain/WhatsApp Image 2026-10-21 at 19.13.39.jpeg")),
        1 / 255.0,
        (imgsz, imgsz),
        swapRB=True,
        crop=False
    )
    net.setInput(dummy_blob)

    t0 = time.time()
    out = net.forward()
    latency_ms = (time.time() - t0) * 1000
    print(f"OpenCV DNN Validation Success: Output shape {out.shape} | Latency: {latency_ms:.2f} ms")
    print("Classes:")
    for i, name in enumerate(["payload_target", "traffic_cone", "landing_circle", "drone_shadow"]):
        print(f"  {i}: {name}")
    print("=" * 60)
    print("Model ready for local folder batching and Raspberry Pi deployment!")
    print("=" * 60)
    return output_onnx


if __name__ == "__main__":
    train_and_export()
