import os
from ultralytics import YOLO


DATA_CFG = os.path.join("data", "hotdog.yaml")
RUNS_ROOT = os.path.join("runs", "hotdog")
TRAIN_NAME = "yolov8m"
VAL_NAME = "yolov8m-val"
PREDS_NAME = "yolov8m-preds"


def main():
    # Hyperparameters via env (fallback to sensible defaults)
    epochs = int(os.environ.get("EPOCHS", 50))
    batch = int(os.environ.get("BATCH", 16))
    imgsz = int(os.environ.get("IMGSZ", 640))
    workers = int(os.environ.get("WORKERS", 2))
    device = os.environ.get("DEVICE", "0")  # Default to GPU 0, set DEVICE=cpu to force CPU

    # Load pretrained YOLOv8m for transfer learning
    model = YOLO("yolov8m.pt")

    # Train
    train_kwargs = dict(
        data=DATA_CFG,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=42,
        workers=workers,           # Windows-friendly
        project=RUNS_ROOT,
        name=TRAIN_NAME,
        exist_ok=True,
        device=device,
        # Common augmentations
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
        # Training niceties
        patience=20,         # early stopping patience
        cos_lr=True,
    )
    if device:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)

    # Validate (computes mAP@0.5 and mAP@0.5:0.95)
    val_kwargs = dict(
        data=DATA_CFG,
        imgsz=imgsz,
        project=RUNS_ROOT,
        name=VAL_NAME,
        exist_ok=True,
        device=device,
    )
    if device:
        val_kwargs["device"] = device
    metrics = model.val(**val_kwargs)

    # Display essential object detection metrics only
    print("\n" + "="*50)
    print("🎯 HOTDOG DETECTION RESULTS")
    print("="*50)
    try:
        print(f"mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
        print(f"Precision    : {metrics.box.mp:.4f}")
        print(f"Recall       : {metrics.box.mr:.4f}")
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
            print(f"Per-class mAP: {metrics.box.maps[0]:.4f} (hotdog)")
    except Exception as e:
        print(f"Metrics available in: {os.path.join(RUNS_ROOT, TRAIN_NAME, 'results.csv')}")
    print("="*50)

    # Inference on validation images to visualize predictions
    pred_kwargs = dict(
        source=os.path.join("Hot Dog Detection YOLO", "valid", "images"),
        imgsz=imgsz,
        conf=0.25,
        iou=0.6,
        save=True,
        project=RUNS_ROOT,
        name=PREDS_NAME,
        exist_ok=True,
        device=device,
    )
    if device:
        pred_kwargs["device"] = device
    model.predict(**pred_kwargs)

    print(f"\n📁 Results saved to: {os.path.join(RUNS_ROOT, TRAIN_NAME)}")
    print("="*50)


if __name__ == "__main__":
    main()
