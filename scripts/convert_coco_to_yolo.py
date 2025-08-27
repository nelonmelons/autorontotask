"""
COCO -> YOLO label converter for the Hot Dog dataset.
- Reads _annotations.coco.json under Hot Dog Detection COCO/{train,valid}
- Writes YOLO labels into Hot Dog Detection YOLO/{train,valid}/labels
- Does not copy images (your YOLO images already exist). Adjust if needed.
"""
import os
import json

COCO_ROOT = os.path.join("Hot Dog Detection COCO")
YOLO_ROOT = os.path.join("Hot Dog Detection YOLO")
SPLITS = ["train", "valid"]
DEFAULT_CLASS_ID = 0  # single class: hotdog


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    # Clamp to [0,1]
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w_n = max(0.0, min(1.0, w_n))
    h_n = max(0.0, min(1.0, h_n))
    return x_c, y_c, w_n, h_n


def main():
    for split in SPLITS:
        coco_json = os.path.join(COCO_ROOT, split, "_annotations.coco.json")
        if not os.path.exists(coco_json):
            print(f"[WARN] Missing {coco_json}, skipping {split}")
            continue

        with open(coco_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        imgs = {img["id"]: img for img in data.get("images", [])}
        anns_by_img = {}
        for ann in data.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        out_lbl_dir = os.path.join(YOLO_ROOT, split, "labels")
        os.makedirs(out_lbl_dir, exist_ok=True)

        written = 0
        for img_id, img in imgs.items():
            img_w, img_h = img["width"], img["height"]
            file_name = img["file_name"]
            stem, _ = os.path.splitext(file_name)
            yolo_txt = os.path.join(out_lbl_dir, f"{stem}.txt")

            lines = []
            for ann in anns_by_img.get(img_id, []):
                if "bbox" not in ann:
                    continue
                x_c, y_c, w_n, h_n = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                lines.append(f"{DEFAULT_CLASS_ID} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

            with open(yolo_txt, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines))
            written += 1

        print(f"[{split}] Wrote labels for {written} images -> {out_lbl_dir}")


if __name__ == "__main__":
    main()
