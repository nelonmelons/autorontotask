import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "Hot Dog Detection YOLO"


def scan_split(split: str):
    img_dir = YOLO / split / "images"
    lbl_dir = YOLO / split / "labels"
    imgs = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    lbls = sorted(lbl_dir.glob("*.txt"))

    img_stems = {p.stem for p in imgs}
    lbl_stems = {p.stem for p in lbls}

    missing_labels = sorted(img_stems - lbl_stems)
    orphan_labels = sorted(lbl_stems - img_stems)

    return {
        "split": split,
        "images": len(imgs),
        "labels": len(lbls),
        "missing_labels": missing_labels[:10],  # show a few
        "orphan_labels": orphan_labels[:10],    # show a few
    }


def main():
    for split in ("train", "valid"):
        stats = scan_split(split)
        print(f"[{split}] images={stats['images']} labels={stats['labels']}")
        if stats["missing_labels"]:
            print(f"  Missing labels for {len(stats['missing_labels'])} images (showing up to 10): {stats['missing_labels']}")
        if stats["orphan_labels"]:
            print(f"  Orphan label files for {len(stats['orphan_labels'])} (showing up to 10): {stats['orphan_labels']}")


if __name__ == "__main__":
    main()
