"""
explore_carr.py

Gustavo Ojeda -- Predictive Assistance Alerts: first look at the CARR dataset
(Customer Activity Recognition in Retail). Counts images and labels per class,
checks that every image has a label, and reports how many people each label file
describes and which class ids it uses.
"""

import argparse
import cv2
import numpy as np
import random
import statistics
from collections import Counter
from pathlib import Path
from ultralytics import YOLO


DEFAULT_ROOT = Path("data/carr/Images and Label/data")


def label_stats(class_dir: Path) -> dict:
    """Count images, labels, missing labels, boxes per file and class ids for one class folder."""
    images = {p.stem for p in (class_dir / "images").glob("*.jpg")}
    labels = {p.stem: p for p in (class_dir / "labels").glob("*.txt")}
    boxes_per_file = Counter()
    class_ids = Counter()
    for path in labels.values():
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        boxes_per_file[len(lines)] += 1
        for line in lines:
            class_ids[int(line.split()[0])] += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "images_without_label": len(images - labels.keys()),
        "boxes_per_file": dict(sorted(boxes_per_file.items())),
        "class_ids": dict(sorted(class_ids.items())),
    }


def yolo_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box (x_center, y_center, w, h) to pixel corners (x1, y1, x2, y2)."""
    x_center, y_center, box_w, box_h = box
    x1 = int((x_center - box_w / 2) * width)
    y1 = int((y_center - box_h / 2) * height)
    x2 = int((x_center + box_w / 2) * width)
    y2 = int((y_center + box_h / 2) * height)
    return x1, y1, x2, y2


def show_sequence(class_dir: Path, start: int, count: int = 6) -> None:
    """Show `count` images with consecutive numbers side by side, with their labeled box drawn."""
    tiles = []
    for index in range(start, start + count):
        name = f"{class_dir.name} ({index})"
        image = cv2.imread(str(class_dir / "images" / f"{name}.jpg"))
        if image is None:
            continue
        values = (class_dir / "labels" / f"{name}.txt").read_text().split()
        box = [float(v) for v in values[1:5]]
        height, width = image.shape[:2]
        x1, y1, x2, y2 = yolo_to_pixels(box, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, str(index), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        tiles.append(cv2.resize(image, (320, 180)))
    print(f"Image size: {width}x{height}")
    title = f"{class_dir.name} {start}-{start + count - 1}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1200, 180)
    cv2.imshow(title, np.hstack(tiles))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Gustavo Ojeda -- Predictive Assistance Alerts: box helpers shared by IoU and overlap ratio
def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def iou(a, b):
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


# Gustavo Ojeda -- Predictive Assistance Alerts: overlap relative to the smaller box,
# so a small box inside a big one counts as the same person
def overlap_ratio(a, b):
    smaller = min(box_area(a), box_area(b))
    return intersection_area(a, b) / smaller if smaller > 0 else 0.0


def pose_check(root: Path, model_path: str, per_class: int, min_iou: float = 0.5) -> None:
    """Run the pose model on a random sample per class and report how often it finds the labeled person."""
    model = YOLO(model_path)
    random.seed(0)
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted((class_dir / "images").glob("*.jpg"))
        sample = random.sample(images, per_class)
        found = 0
        conf_total = 0.0
        for image_path in sample:
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            values = (class_dir / "labels" / f"{image_path.stem}.txt").read_text().split()
            label_box = yolo_to_pixels([float(v) for v in values[1:5]], width, height)
            result = model(image, verbose=False)[0]
            best_iou, best_index = 0.0, None
            for index, box in enumerate(result.boxes.xyxy.tolist()):
                overlap = iou(label_box, box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_iou >= min_iou:
                found += 1
                conf_total += float(result.keypoints.conf[best_index].mean())
        mean_conf = conf_total / found if found else 0.0
        print(f"{class_dir.name}: found {found}/{per_class} ({found / per_class:.0%}), "
              f"mean keypoint confidence {mean_conf:.2f}")


def frame_number(path: Path) -> int:
    """Extract N from a file named '<Class> (N).txt'."""
    return int(path.stem.rsplit("(", 1)[1].rstrip(")"))

# Gustavo Ojeda -- Predictive Assistance Alerts: use overlap ratio so label boxes that switch
# between full body and arm only do not create false clip boundaries
def segment_clips(class_dir, min_overlap=0.5):
    """Split a class's frames into clips: a new clip starts when the frame number skips or the labeled box jumps."""
    boxes = {}
    for path in (class_dir / "labels").glob("*.txt"):
        values = path.read_text().split()
        # All CARR images are 1280x720, and overlap ratio does not depend on the scale anyway.
        boxes[frame_number(path)] = yolo_to_pixels([float(v) for v in values[1:5]], 1280, 720)
    clips = []
    previous = None
    for number in sorted(boxes):
        if previous is None or number != previous + 1 or overlap_ratio(boxes[previous], boxes[number]) < min_overlap:
            clips.append([])
        clips[-1].append(number)
        previous = number
    return clips


def clip_report(root: Path) -> None:
    """Print how many clips each class has and how long they are."""
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lengths = sorted(len(clip) for clip in segment_clips(class_dir))
        long_clips = sum(1 for n in lengths if n >= 30)
        print(f"{class_dir.name}: {len(lengths)} clips, median {statistics.median(lengths)} frames, "
              f"min {lengths[0]}, max {lengths[-1]}, clips with 30+ frames: {long_clips}")


def main() -> None:
    """Print label statistics for every class folder."""
    parser = argparse.ArgumentParser(description="Explore the CARR dataset")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder with one subfolder per class")
    parser.add_argument("--show", default=None, help="Class folder name to view, e.g. 'Viewing'")
    parser.add_argument("--start", type=int, default=1, help="First image number to show (default: 1)")
    parser.add_argument("--pose-check", type=int, default=0, help="Run the pose model on N random images per class")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Pose model weights (default: yolov8n-pose.pt)")
    parser.add_argument("--clips", action="store_true", help="Report how the frames split into video clips")
    parser.add_argument("--clip-list", default=None, help="Class folder name: list its first clips (start frame and length)")
    args = parser.parse_args()
    if args.clip_list:
        for clip in segment_clips(args.root / args.clip_list)[:15]:
            print(f"start {clip[0]}, length {len(clip)}")
        return

    if args.clips:
        clip_report(args.root)
        return

    if args.show:
        show_sequence(args.root / args.show, args.start)
        return

    if args.pose_check:
        pose_check(args.root, args.model, args.pose_check)
        return

    for class_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        print(f"{class_dir.name}:")
        for key, value in label_stats(class_dir).items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()