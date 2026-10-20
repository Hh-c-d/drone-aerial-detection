"""
Train a YOLO model for drone aerial object detection (VisDrone-style data).

Dataset layout (YOLO format):
    dataset/
        images/train/*.jpg   labels/train/*.txt
        images/val/*.jpg     labels/val/*.txt
    data.yaml

Usage:
    python train.py --data data.yaml --model yolo11s.pt --epochs 100
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO on aerial imagery")
    p.add_argument("--data", default="data.yaml", help="dataset config (YOLO format)")
    p.add_argument("--model", default="yolo11s.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. '0' or 'cpu'; default = auto")
    p.add_argument("--project", default="runs/aerial")
    p.add_argument("--name", default="train")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
