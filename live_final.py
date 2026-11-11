"""
Live FPV / Webcam Object Detection Tester.
Optimized for local testing on live video feeds (FPV USB receiver, webcam, video file).
Uses the tailored ONNX model via OpenCV DNN for zero-latency real-time inference.

Controls:
  [q] Quit
  [s] Save snapshot (annotated frame + crops + telemetry)
  [f] Toggle drone shadow filter
  [c] Cycle confidence threshold (0.20 -> 0.30 -> 0.40 -> 0.50)
"""

import argparse
import csv
import os
import time

# Suppress Qt Wayland font plugin warnings on Linux desktops
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from batch_detect import TerrainDetectorONNX, draw_hud


def make_snapshot_dir(base_dir="outputs/live_snapshots"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("snap_")]
    idx = len(existing) + 1
    snap_dir = os.path.join(base_dir, f"snap_{idx:03d}")
    os.makedirs(snap_dir, exist_ok=True)
    return snap_dir


def main():
    parser = argparse.ArgumentParser(description="Live FPV Terrain Object Detector")
    parser.add_argument("--source", default="0", help="Camera index (0, 1), video file, or RTSP URL")
    parser.add_argument("--model", default="mars_terrain_nano.onnx", help="Path to tailored ONNX model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=640, help="Resize input width for speed")
    parser.add_argument("--filter-shadows", action="store_true", help="Filter quadcopter shadows")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    # Try setting 640x480 resolution for high FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    conf_levels = [0.20, 0.30, 0.40, 0.50]
    conf_idx = 0
    filter_shadows = args.filter_shadows

    detector = TerrainDetectorONNX(
        onnx_model_path=args.model,
        conf_threshold=conf_levels[conf_idx],
    )

    print("\n" + "=" * 60)
    print("Live FPV Detector Started!")
    print(f" Source: {args.source}")
    print(f" Model:  {args.model}")
    print(" Keys:   [q] = quit | [s] = save snapshot | [f] = toggle shadow | [c] = change conf")
    print("=" * 60)

    fps_smooth = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame could not be read.")
            break

        frame_count += 1
        h, w = frame.shape[:2]
        if w > args.width:
            scale = args.width / float(w)
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        detector.conf_threshold = conf_levels[conf_idx]
        detections, latency_ms = detector.detect(frame)

        if filter_shadows:
            detections = [d for d in detections if d["label"] != "drone_shadow"]

        fps_instant = 1000.0 / max(latency_ms, 0.001)
        fps_smooth = fps_instant if fps_smooth == 0.0 else (0.85 * fps_smooth + 0.15 * fps_instant)

        display_view = draw_hud(frame, detections, latency_ms, f"LIVE FPV | Conf: {conf_levels[conf_idx]:.2f}")
        cv2.imshow("Live FPV Object Detection", display_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            filter_shadows = not filter_shadows
            print(f"[INFO] Drone shadow filter: {'ENABLED' if filter_shadows else 'DISABLED'}")
        elif key == ord("c"):
            conf_idx = (conf_idx + 1) % len(conf_levels)
            print(f"[INFO] Confidence threshold switched to: {conf_levels[conf_idx]:.2f}")
        elif key == ord("s"):
            snap_dir = make_snapshot_dir()
            cv2.imwrite(os.path.join(snap_dir, "raw_frame.jpg"), frame)
            cv2.imwrite(os.path.join(snap_dir, "annotated_frame.jpg"), display_view)

            crops_dir = os.path.join(snap_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            for d in detections:
                x, y, bw, bh = d["box"]
                crop = frame[y:y + bh, x:x + bw]
                c_sub = os.path.join(crops_dir, d["label"])
                os.makedirs(c_sub, exist_ok=True)
                cv2.imwrite(os.path.join(c_sub, f"obj_{d['id']:02d}_{d['score']:.2f}.jpg"), crop)

            with open(os.path.join(snap_dir, "detections.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "label", "confidence", "x", "y", "width", "height", "area"])
                for d in detections:
                    x, y, bw, bh = d["box"]
                    writer.writerow([d["id"], d["label"], round(d["score"], 4), x, y, bw, bh, d["area"]])

            print(f"[SNAPSHOT] Saved {len(detections)} detections to {snap_dir}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
