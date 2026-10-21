# Gustavo Ojeda -- Predictive Assistance Alerts: visual quality check of extracted CARR keypoints
"""Draw saved skeletons on random CARR images to check they land on the labeled person.

Green box = label from the dataset. Blue box = person matched by YOLO-pose.
Red dots and lines = saved keypoints (only points with enough confidence).
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.explore_carr import DEFAULT_ROOT, yolo_to_pixels

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def sample_rows(keypoints_dir, status, per_class, rng):
    """Pick a few random rows with the given status from each class CSV."""
    chosen = []
    for path in sorted(keypoints_dir.glob("*.csv")):
        with open(path, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["status"] == status]
        chosen += rng.sample(rows, min(per_class, len(rows)))
    return chosen


def draw_row(root, row, min_conf):
    class_dir = root / row["class"]
    label_path = next((class_dir / "labels").glob(f"* ({row['frame']}).txt"))
    image = cv2.imread(str(class_dir / "images" / f"{label_path.stem}.jpg"))
    height, width = image.shape[:2]
    values = label_path.read_text().split()
    x1, y1, x2, y2 = [int(v) for v in yolo_to_pixels([float(v) for v in values[1:5]], width, height)]
    if row["status"] == "ok":
        bx1, by1, bx2, by2 = [int(float(row[k])) for k in ("box_x1", "box_y1", "box_x2", "box_y2")]
        cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        points = []
        for i in range(17):
            x, y, c = float(row[f"kp{i}_x"]), float(row[f"kp{i}_y"]), float(row[f"kp{i}_conf"])
            points.append((int(x), int(y)) if c >= min_conf else None)
        for a, b in SKELETON:
            if points[a] and points[b]:
                cv2.line(image, points[a], points[b], (0, 0, 255), 2)
        for point in points:
            if point:
                cv2.circle(image, point, 4, (0, 0, 255), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.putText(image, f"{row['class']} {row['frame']}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return cv2.resize(image, (480, 270))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--keypoints", type=Path, default=Path("data/carr/keypoints"))
    parser.add_argument("--status", default="ok")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = sample_rows(args.keypoints, args.status, args.per_class, rng)
    tiles = [draw_row(args.root, row, args.min_conf) for row in rows]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.namedWindow("keypoint check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("keypoint check", 1440, 610)
    cv2.imshow("keypoint check", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()