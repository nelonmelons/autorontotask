# Hot Dog Detection with Ultralytics YOLOv8m

End-to-end training and evaluation of a single-class (hotdog) detector with YOLOv8m, optimized for GPU training.

## Dataset

- YOLO format (already in repo): `Hot Dog Detection YOLO/train|valid/{images,labels}`
- Optional COCO: `Hot Dog Detection COCO/{train,valid}/_annotations.coco.json`

The data config is in `data/hotdog.yaml`.

## Setup (PowerShell)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Train + Validate + Predict (GPU-enabled)

```powershell
python .\scripts\train_yolov8m.py
```

Artifacts are saved under `runs/hotdog/`:

- Training: `runs/hotdog/yolov8m/`
- Validation (mAP): `runs/hotdog/yolov8m-val/`
- Predictions on val images: `runs/hotdog/yolov8m-preds/`

### Environment Variables

Customize training via environment variables:

```powershell
$env:EPOCHS=50      # Number of training epochs (default: 50)
$env:BATCH=16       # Batch size (default: 16)
$env:IMGSZ=640      # Image size (default: 640)
$env:WORKERS=2      # Data loader workers (default: 2)
$env:DEVICE="0"     # GPU device (default: "0", use "cpu" for CPU-only)
python .\scripts\train_yolov8m.py
```

## GPU Requirements

- CUDA-compatible GPU (tested on RTX 2070 SUPER)
- CUDA 11.8 runtime (installed via PyTorch)
- ~4GB GPU memory for batch=8, imgsz=640

## Optional: Regenerate YOLO labels from COCO

Only needed if you want to re-create YOLO labels from the COCO JSONs.

```powershell
python .\scripts\convert_coco_to_yolo.py
```

## Tips

- Bump epochs to 100–150 for higher accuracy.
- Scale the model: `yolov8s.pt` (faster) < `yolov8m.pt` (balanced) < `yolov8l.pt` (more accurate).
- Adjust `BATCH` and `IMGSZ` based on your GPU memory.
- For CPU training, set `$env:DEVICE="cpu"`.

## Results

Quick 1-epoch smoke test completed successfully:

- GPU: NVIDIA GeForce RTX 2070 SUPER
- Training speed: ~12.6ms inference per image at 320x320
- mAP metrics computed and saved to validation artifacts
