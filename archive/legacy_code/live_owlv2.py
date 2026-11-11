import argparse
import csv
import os
import re
import threading
import time
import traceback

import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL_ID = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["a crashed spacecraft backshell", "a rock", "a person", "a spectacle"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_run_folder(base_dir="outputs", prefix="snap_"):
    os.makedirs(base_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    ids = [int(m.group(1)) for n in os.listdir(base_dir)
           if (m := pat.match(n))]
    d = os.path.join(base_dir, f"{prefix}{max(ids, default=0) + 1:03d}")
    os.makedirs(d)
    return d


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)


def nms(dets, thr=0.5):
    kept = []
    for c in sorted(dets, key=lambda d: d["score"], reverse=True):
        if all(iou(c["box"], k["box"]) < thr for k in kept):
            kept.append(c)
    for n, d in enumerate(kept, 1):
        d["id"] = n
    return kept


def to_pixel_box(b, side, w, h):
    """Normalized (cx, cy, w, h) on the padded square -> clipped (x, y, w, h)."""
    cx, cy, bw, bh = b
    x1 = max(0, int((cx - bw / 2) * side))
    y1 = max(0, int((cy - bh / 2) * side))
    x2 = min(w, int((cx + bw / 2) * side))
    y2 = min(h, int((cy + bh / 2) * side))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


class Detector:
    def __init__(self, prompts, threshold):
        self.prompts = prompts
        self.threshold = threshold
        self.proc = Owlv2Processor.from_pretrained(MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID)
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, frame_bgr):
        """Returns (detections above threshold, top-3 raw boxes, best score)."""
        h, w = frame_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.proc(text=[self.prompts], images=pil,
                           return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)

        side = max(h, w)  # OWLv2 pads to a square (bottom/right)
        scores, labels = out.logits[0].sigmoid().max(dim=-1)
        boxes = out.pred_boxes[0].cpu()
        scores, labels = scores.cpu(), labels.cpu()

        dets = []
        for b, s, l in zip(boxes, scores, labels):
            if s < self.threshold:
                continue
            box = to_pixel_box(b.tolist(), side, w, h)
            if box is None:
                continue
            dets.append({"box": box, "score": float(s),
                         "area": box[2] * box[3],
                         "label": self.prompts[int(l)]})

        # Top raw boxes regardless of threshold, for debugging.
        weak = []
        for i in torch.topk(scores, k=min(3, len(scores))).indices.tolist():
            box = to_pixel_box(boxes[i].tolist(), side, w, h)
            if box is not None:
                weak.append({"box": box, "score": float(scores[i])})

        return nms(dets), weak, float(scores.max())


class Worker(threading.Thread):
    """Runs detection on the newest frame, skipping frames it can't keep up with."""

    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.lock = threading.Lock()
        self.frame = None
        self.dets = []
        self.weak = []
        self.best = 0.0
        self.det_frame = None   # the frame the current dets belong to
        self.latency = 0.0
        self.passes = 0
        self.running = True

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def snapshot(self):
        with self.lock:
            return (list(self.dets), list(self.weak), self.best,
                    self.det_frame, self.latency, self.passes)

    def run(self):
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                t0 = time.time()
                dets, weak, best = self.detector(frame)
                with self.lock:
                    self.dets, self.weak, self.best = dets, weak, best
                    self.det_frame = frame
                    self.latency = time.time() - t0
                    self.passes += 1
            except Exception:
                traceback.print_exc()
                time.sleep(1)


def draw(frame, dets, weak=None):
    out = frame.copy()
    if weak:
        for d in weak:
            x, y, bw, bh = d["box"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 255), 1)
            cv2.putText(out, f"{d['score']:.2f}", (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)
    for d in dets:
        x, y, bw, bh = d["box"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        tag = f"#{d['id']} {d['label'].split()[-1]} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty), (0, 255, 0), -1)
        cv2.putText(out, tag, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_snapshot(base_dir, frame, dets):
    run_dir = make_run_folder(base_dir)
    cv2.imwrite(os.path.join(run_dir, "frame.jpg"), frame)
    cv2.imwrite(os.path.join(run_dir, "detections.jpg"), draw(frame, dets))
    crop_dir = os.path.join(run_dir, "crops")
    os.makedirs(crop_dir)
    with open(os.path.join(run_dir, "detections.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label", "x", "y", "width", "height", "area", "score"])
        for d in dets:
            x, y, bw, bh = d["box"]
            cv2.imwrite(os.path.join(crop_dir, f"object_{d['id']:03d}.jpg"),
                        frame[y:y + bh, x:x + bw])
            wr.writerow([d["id"], d["label"], x, y, bw, bh, d["area"],
                         round(d["score"], 4)])
    print(f"Saved {run_dir} ({len(dets)} detections)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="camera index, video file, or RTSP/HTTP URL")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--prompts", nargs="+", default=PROMPTS)
    p.add_argument("--width", type=int, default=960,
                   help="resize frames to this width before processing")
    p.add_argument("--debug", action="store_true",
                   help="draw the top 3 raw boxes in yellow even below threshold")
    a = p.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {a.source}")

    print(f"Loading model on {DEVICE}...")
    worker = Worker(Detector(a.prompts, a.threshold))
    worker.start()

    print("Keys: q = quit, s = save snapshot")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > a.width:
            scale = a.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        worker.submit(frame)
        dets, weak, best, det_frame, latency, passes = worker.snapshot()

        view = draw(frame, dets, weak if a.debug else None)
        status = (f"{len(dets)} obj | best {best:.2f} | "
                  f"{latency:.2f}s/frame | passes {passes}")
        if passes == 0:
            status = "waiting for first detection pass..."
        cv2.putText(view, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Live detection", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and det_frame is not None:
            save_snapshot(a.output_dir, det_frame, dets)

    worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()