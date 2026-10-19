# Gustavo Ojeda -- Predictive Assistance Alerts: extract YOLO-pose keypoints for every CARR frame
"""Run YOLO-pose on CARR images and save the 17 keypoints of the labeled person.

A detected person qualifies if it covers most of the labeled box (overlap ratio,
because labels sometimes cover only an arm); among those, the most similar box
(IoU) wins. Every frame gets a status, so data quality can be measured later.
Each class is written to its own CSV, so an interrupted run can be resumed.
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from scripts.explore_carr import DEFAULT_ROOT, frame_number, iou, overlap_ratio, yolo_to_pixels

KEYPOINT_COLUMNS = [f"kp{i}_{axis}" for i in range(17) for axis in ("x", "y", "conf")]
HEADER = ["class", "frame", "split", "status", "box_x1", "box_y1", "box_x2", "box_y2"] + KEYPOINT_COLUMNS


def read_split(path):
    """Return {(class_name, frame): split} from split.csv."""
    with open(path, newline="") as f:
        return {(row["class"], int(row["frame"])): row["split"] for row in csv.DictReader(f)}


def best_match(boxes, label_box, min_overlap):
    """Index of the detected person that matches the label box, or None."""
    best_index, best_iou = None, -1.0
    for index, box in enumerate(boxes):
        if overlap_ratio(box, label_box) < min_overlap:
            continue
        score = iou(box, label_box)
        if score > best_iou:
            best_index, best_iou = index, score
    return best_index


def process_frame(model, image_path, label_path, conf, min_overlap):
    """Return (status, values) where values = person box + 51 keypoint numbers."""
    image = cv2.imread(str(image_path))
    if image is None:
        return "no_image", []
    height, width = image.shape[:2]
    values = label_path.read_text().split()
    label_box = yolo_to_pixels([float(v) for v in values[1:5]], width, height)
    result = model(image, conf=conf, verbose=False)[0]
    if len(result.boxes) == 0:
        return "no_person", []
    index = best_match(result.boxes.xyxy.tolist(), label_box, min_overlap)
    if index is None:
        return "no_match", []
    box = [round(v, 1) for v in result.boxes.xyxy[index].tolist()]
    keypoints = []
    for (x, y), c in zip(result.keypoints.xy[index].tolist(), result.keypoints.conf[index].tolist()):
        keypoints += [round(x, 1), round(y, 1), round(c, 3)]
    return "ok", box + keypoints


def extract_class(class_dir, split, model, out_path, args):
    labels = {frame_number(p): p for p in (class_dir / "labels").glob("*.txt")}
    frames = sorted(n for (name, n) in split if name == class_dir.name)
    if args.limit:
        frames = frames[:args.limit]
    counts = Counter()
    start = time.time()
    part_path = out_path.with_suffix(".part")
    with open(part_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for i, number in enumerate(frames, start=1):
            label_path = labels[number]
            image_path = class_dir / "images" / f"{label_path.stem}.jpg"
            status, values = process_frame(model, image_path, label_path, args.conf, args.min_overlap)
            counts[status] += 1
            writer.writerow([class_dir.name, number, split[(class_dir.name, number)], status] + values)
            if i % 1000 == 0:
                print(f"  {class_dir.name}: {i}/{len(frames)} frames, {i / (time.time() - start):.1f} fps")
    part_path.replace(out_path)
    print(f"{class_dir.name}: {dict(counts)}, ok {counts['ok'] / len(frames):.1%}, "
          f"{len(frames) / (time.time() - start):.1f} fps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", type=Path, default=Path("data/carr/split.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/carr/keypoints"))
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="only the first N frames per class (quick test)")
    args = parser.parse_args()

    split = read_split(args.split)
    model = YOLO(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        out_path = args.out / f"{class_dir.name}.csv"
        if out_path.exists():
            print(f"{class_dir.name}: already done, skipping")
            continue
        extract_class(class_dir, split, model, out_path, args)


if __name__ == "__main__":
    main()