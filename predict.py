"""
Inference on aerial images, with optional tiled (sliced) inference.

Small objects are the classic failure mode of drone footage: a 640px model
input downsamples a 4K frame until a person is a few pixels wide. Tiled
inference runs the model on overlapping crops at near-native resolution and
merges the boxes with NMS -- typically a big mAP boost on VisDrone.

Usage:
    python predict.py --weights best.pt --source test.jpg
    python predict.py --weights best.pt --source images/ --slice 640 --overlap 0.2
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from ultralytics import YOLO


def tile_starts(size, tile, overlap):
    """Start offsets so tiles cover [0, size] with a flush last tile."""
    if size <= tile:
        return [0]
    step = max(int(tile * (1 - overlap)), 1)
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def sliced_predict(model, img, tile=640, overlap=0.2, conf=0.25, iou=0.5):
    """Run tiled inference, return list of (box_xyxy, score, class_id)."""
    boxes, scores, clses = [], [], []
    for ys in tile_starts(img.height, tile, overlap):
        for xs in tile_starts(img.width, tile, overlap):
            crop = img.crop((xs, ys, min(xs + tile, img.width), min(ys + tile, img.height)))
            r = model.predict(crop, conf=conf, verbose=False)[0]
            if len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            b[:, [0, 2]] += xs
            b[:, [1, 3]] += ys
            boxes.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
            clses.append(r.boxes.cls.cpu().numpy())
    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    clses = np.concatenate(clses)
    keep = nms(torch.tensor(boxes), torch.tensor(scores), iou).cpu().numpy()
    return [(boxes[i], float(scores[i]), int(clses[i])) for i in keep]


def draw(img, preds, names):
    d = ImageDraw.Draw(img)
    for box, score, cls in preds:
        x1, y1, x2, y2 = box
        d.rectangle([x1, y1, x2, y2], outline="red", width=2)
        d.text((x1, max(y1 - 12, 0)), f"{names[cls]} {score:.2f}", fill="red")
    return img


def main():
    p = argparse.ArgumentParser(description="Aerial detection inference")
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--source", required=True, help="image, directory of images, or glob")
    p.add_argument("--out", default="runs/aerial/predict", help="output directory")
    p.add_argument("--slice", type=int, default=0,
                   help="tile size for sliced inference (0 = whole-image)")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.weights)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    imgs = [src] if src.is_file() else sorted(src.rglob("*.jpg")) + sorted(src.rglob("*.png"))

    for f in imgs:
        img = Image.open(f).convert("RGB")
        if args.slice:
            preds = sliced_predict(model, img, tile=args.slice,
                                   overlap=args.overlap, conf=args.conf)
            draw(img, preds, model.names)
            print(f"{f.name}: {len(preds)} detections (sliced)")
        else:
            r = model.predict(img, conf=args.conf, verbose=False)[0]
            img = Image.fromarray(r.plot()[:, :, ::-1])
            print(f"{f.name}: {len(r.boxes)} detections")
        img.save(outdir / f"{f.stem}_pred.jpg")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
