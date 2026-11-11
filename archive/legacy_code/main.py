import argparse
import csv
import os
import re

import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL_ID = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["a crashed spacecraft backshell", "a parachute", "a rock"]


def make_run_folder(base_dir="outputs", prefix="output_"):
    os.makedirs(base_dir, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    ids = [int(m.group(1)) for n in os.listdir(base_dir)
           if (m := pat.match(n))]
    run_dir = os.path.join(base_dir, f"{prefix}{max(ids, default=0) + 1:03d}")
    os.makedirs(run_dir)
    return run_dir


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)


def detect_owl(image_bgr, prompts, threshold=0.2):
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval()

    h, w = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(text=[prompts], images=pil, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    # OWLv2 pads the image to a square, so rescale against the padded size
    side = max(h, w)
    res = processor.post_process_grounded_object_detection(
        out,
        threshold=threshold,
        target_sizes=torch.tensor([[side, side]]),
        text_labels=[prompts],
    )[0]

    dets = []
    for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
        x1, y1, x2, y2 = box.tolist()
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append({
            "box": (x1, y1, x2 - x1, y2 - y1),
            "score": float(score),
            "area": (x2 - x1) * (y2 - y1),
            "label": label,
        })
    return dets


def nms(dets, iou_thr=0.5):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for c in dets:
        if all(iou(c["box"], k["box"]) < iou_thr for k in kept):
            kept.append(c)
    return kept


def run(image_path, output_dir="outputs", prompts=PROMPTS,
        threshold=0.2, save_crops=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    run_dir = make_run_folder(output_dir)
    kept = nms(detect_owl(image, prompts, threshold))
    for n, d in enumerate(kept, start=1):
        d["id"] = n

    result = image.copy()
    for d in kept:
        x, y, bw, bh = d["box"]
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        tag = f"#{d['id']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y)
        cv2.rectangle(result, (x, ty - th - 4), (x + tw + 4, ty), (0, 255, 0), -1)
        cv2.putText(result, tag, (x + 2, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(run_dir, "detections.jpg"), result)

    if save_crops:
        crop_dir = os.path.join(run_dir, "crops")
        os.makedirs(crop_dir)
        for d in kept:
            x, y, bw, bh = d["box"]
            cv2.imwrite(os.path.join(crop_dir, f"object_{d['id']:03d}.jpg"),
                        image[y:y + bh, x:x + bw])

    with open(os.path.join(run_dir, "detections.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label", "x", "y", "width", "height", "area", "score"])
        for d in kept:
            x, y, bw, bh = d["box"]
            wr.writerow([d["id"], d["label"], x, y, bw, bh,
                         d["area"], round(d["score"], 4)])

    print(f"Output folder: {run_dir}")
    print(f"Detections: {len(kept)}")
    for d in kept:
        print(f"{d['id']:>3}  {d['label']:<32} {str(d['box']):<24} {d['score']:.3f}")
    return kept, run_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--prompts", nargs="+", default=PROMPTS)
    p.add_argument("--no-crops", action="store_true")
    a = p.parse_args()
    run(a.image, a.output_dir, a.prompts, a.threshold, not a.no_crops)