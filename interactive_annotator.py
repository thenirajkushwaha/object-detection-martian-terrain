"""
Interactive Image Annotation Tool for Terrain Detection.
Allows the user to click-and-drag bounding boxes on images from a folder,
assign labels ('object', 'cone', 'landing_mark'), and trigger ONNX generation.

Classes:
  1: object       (payload target, equipment, or obstacle)
  2: cone         (traffic cone / marker)
  3: landing_mark (landing circle, chalk ring, or helipad mark)

Controls:
  - Mouse Click & Drag : Draw bounding box
  - Number keys [1, 2, 3] : Switch active class
  - [u] : Undo last box on this image
  - [c] : Clear all boxes on this image
  - [d] or [Space] : Next image (saves current annotations)
  - [a] : Previous image
  - [t] : Finish labeling & trigger automatic ONNX model training
  - [q] : Save and exit
"""

import argparse
import glob
import os
import cv2
import numpy as np

DEFAULT_CLASSES = ["object", "cone", "landing_mark"]

CLASS_COLORS = {
    0: (255, 255, 0),    # Cyan/Yellow: object
    1: (0, 140, 255),    # Orange: cone
    2: (0, 255, 128),    # Mint Green: landing_mark
}


class InteractiveAnnotator:
    def __init__(self, image_dir="mars_terrain", labels_dir="datasets/user_labels", classes=None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.classes = classes or DEFAULT_CLASSES
        os.makedirs(self.labels_dir, exist_ok=True)

        # Gather supported images
        patterns = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp"]
        self.image_paths = []
        for pat in patterns:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, pat)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, pat.upper())))
        self.image_paths = sorted(list(set(self.image_paths)))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in '{image_dir}'.")

        self.current_idx = 0
        self.current_boxes = []  # list of [cls_id, x1, y1, x2, y2]
        self.active_class = 0

        # Mouse state
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.window_name = "Terrain Image Labeler"

    def _get_label_path(self, img_path):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, f"{base_name}.txt")

    def load_annotations(self, img_path, img_w, img_h):
        """Loads existing YOLO annotations if available."""
        txt_path = self._get_label_path(img_path)
        boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        x1 = int((cx - w / 2.0) * img_w)
                        y1 = int((cy - h / 2.0) * img_h)
                        x2 = int((cx + w / 2.0) * img_w)
                        y2 = int((cy + h / 2.0) * img_h)
                        boxes.append([cls_id, x1, y1, x2, y2])
        return boxes

    def save_annotations(self, img_path, img_w, img_h):
        """Saves current boxes to YOLO format (.txt)."""
        txt_path = self._get_label_path(img_path)
        with open(txt_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in self.current_boxes:
                # clamp
                x1, x2 = max(0, min(img_w, x1)), max(0, min(img_w, x2))
                y1, y2 = max(0, min(img_h, y1)), max(0, min(img_h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / float(img_w)
                h = (y2 - y1) / float(img_h)
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_pt = (x, y)
                x1, x2 = sorted([self.start_pt[0], self.end_pt[0]])
                y1, y2 = sorted([self.start_pt[1], self.end_pt[1]])
                # Filter tiny accidental clicks
                if (x2 - x1) > 6 and (y2 - y1) > 6:
                    self.current_boxes.append([self.active_class, x1, y1, x2, y2])

    def render_canvas(self, raw_img):
        canvas = raw_img.copy()
        h, w = canvas.shape[:2]

        # Draw existing boxes
        for cls_id, x1, y1, x2, y2 in self.current_boxes:
            col = CLASS_COLORS.get(cls_id, (0, 255, 255))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            label_text = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty), col, -1)
            cv2.putText(canvas, label_text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw active dragging box
        if self.drawing and self.start_pt and self.end_pt:
            col = CLASS_COLORS.get(self.active_class, (0, 255, 255))
            cv2.rectangle(canvas, self.start_pt, self.end_pt, col, 1)

        # Top Information Banner
        banner_h = 48
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        filename = os.path.basename(self.image_paths[self.current_idx])
        txt_info = f"[{self.current_idx + 1}/{len(self.image_paths)}] {filename} | Boxes: {len(self.current_boxes)}"
        cv2.putText(canvas, txt_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Class selectors prompt
        txt_classes = "Active: "
        x_offset = 10
        cv2.putText(canvas, txt_classes, (x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        x_offset += 65

        for i, c_name in enumerate(self.classes):
            col = CLASS_COLORS.get(i, (255, 255, 255))
            is_active = (i == self.active_class)
            c_tag = f"[{i + 1}] {c_name} "
            if is_active:
                cv2.rectangle(canvas, (x_offset - 2, 26), (x_offset + len(c_tag) * 8 + 4, 44), col, -1)
                cv2.putText(canvas, c_tag, (x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, c_tag, (x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            x_offset += len(c_tag) * 8 + 10

        # Footer guide
        footer_y = h - 10
        txt_footer = "[Mouse]: Draw | [1-3]: Class | [u]: Undo | [c]: Clear | [Space]: Next | [t]: Train ONNX | [q]: Quit"
        cv2.putText(canvas, txt_footer, (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        img_path = self.image_paths[self.current_idx]
        raw_img = cv2.imread(img_path)
        h, w = raw_img.shape[:2]
        self.current_boxes = self.load_annotations(img_path, w, h)

        trigger_train = False

        while True:
            display_frame = self.render_canvas(raw_img)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # Class selection keys
            if key in [ord("1"), ord("2"), ord("3")]:
                selected = int(chr(key)) - 1
                if selected < len(self.classes):
                    self.active_class = selected
            # Undo
            elif key in [ord("u"), ord("U")]:
                if self.current_boxes:
                    self.current_boxes.pop()
            # Clear all
            elif key in [ord("c"), ord("C")]:
                self.current_boxes.clear()
            # Next image
            elif key in [ord(" "), ord("d"), ord("D")]:
                self.save_annotations(img_path, w, h)
                if self.current_idx < len(self.image_paths) - 1:
                    self.current_idx += 1
                    img_path = self.image_paths[self.current_idx]
                    raw_img = cv2.imread(img_path)
                    h, w = raw_img.shape[:2]
                    self.current_boxes = self.load_annotations(img_path, w, h)
                else:
                    print("[INFO] Reached last image in folder.")
            # Previous image
            elif key in [ord("a"), ord("A")]:
                self.save_annotations(img_path, w, h)
                if self.current_idx > 0:
                    self.current_idx -= 1
                    img_path = self.image_paths[self.current_idx]
                    raw_img = cv2.imread(img_path)
                    h, w = raw_img.shape[:2]
                    self.current_boxes = self.load_annotations(img_path, w, h)
            # Finish & Train
            elif key in [ord("t"), ord("T")]:
                self.save_annotations(img_path, w, h)
                trigger_train = True
                break
            # Quit
            elif key in [ord("q"), ord("Q"), 27]:
                self.save_annotations(img_path, w, h)
                break

        cv2.destroyAllWindows()
        return trigger_train


def main():
    parser = argparse.ArgumentParser(description="Interactive Image Annotation Tool for Terrain Detection")
    parser.add_argument("-i", "--input-dir", default="mars_terrain", help="Folder containing images to annotate")
    parser.add_argument("-l", "--labels-dir", default="datasets/user_labels", help="Folder to save label .txt files")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="List of class names")
    args = parser.parse_args()

    annotator = InteractiveAnnotator(
        image_dir=args.input_dir,
        labels_dir=args.labels_dir,
        classes=args.classes,
    )
    should_train = annotator.run()

    if should_train:
        print("\n[INFO] Triggering automatic dataset build & ONNX training from your labels...")
        from train_from_annotations import build_and_train_onnx
        build_and_train_onnx(
            images_dir=args.input_dir,
            labels_dir=args.labels_dir,
            classes=args.classes,
            output_onnx="mars_terrain_nano.onnx",
        )
    else:
        print("\n[INFO] Annotations saved successfully. You can run training anytime with 'python train_from_annotations.py'.")


if __name__ == "__main__":
    main()
