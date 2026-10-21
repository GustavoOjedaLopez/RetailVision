# Gustavo Ojeda -- Predictive Assistance Alerts: turn CARR keypoints into short movement windows with features
"""Cut the extracted CARR keypoints into short windows and describe each one with numbers.

A single frame cannot tell "picking" from "returning"; a short piece of movement can.
Each window is WINDOW consecutive frames of the same person. A window never crosses:
  - a gap in frame numbers (frames dropped by the split buffer or by no_match),
  - a change of split (train / val / test),
  - a scene cut (the person's box jumps to a different place between two frames).

Each window becomes one row of features (pose shape, visibility, hand movement,
body movement) plus its class and split. That table is what the Level 1 model learns from.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 60            # frames per window (60 beat 30 on validation, 2026-10-20)
STRIDE = 15            # a new window starts every STRIDE frames (windows overlap by half)
MIN_CONF = 0.3         # keypoints below this confidence are treated as missing
MIN_BOX_OVERLAP = 0.5  # below this, two consecutive boxes are a scene cut

NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
         "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
         "l_knee", "r_knee", "l_ankle", "r_ankle"]
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
BOX_COLUMNS = ["box_x1", "box_y1", "box_x2", "box_y2"]
# Gustavo Ojeda -- Predictive Assistance Alerts: actions a skeleton can hardly tell apart, joined into 4 groups
GROUPS = {
    "No Interest": "not engaged",
    "Viewing": "looking", "Turning to Shelf": "looking",
    "Touching": "touching",
    "Picking And Returning": "picking", "Picking and Putting": "picking",
}


def stat(function, values):
    """Apply a nan-aware statistic, or return NaN when there is no valid value at all."""
    values = values[np.isfinite(values)]
    return function(values) if values.size else np.nan


def overlap_ratio(a, b):
    """Overlap of two lists of boxes (rows of x1, y1, x2, y2), relative to the smaller box."""
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return ix * iy / np.minimum(area_a, area_b)


def segments(frames, splits, boxes):
    """Yield (start, end) row ranges that are continuous movement of one person."""
    cut = ((np.diff(frames) != 1)
           | (splits[1:] != splits[:-1])
           | (overlap_ratio(boxes[1:], boxes[:-1]) < MIN_BOX_OVERLAP))
    edges = np.flatnonzero(np.r_[True, cut, True])
    yield from zip(edges[:-1], edges[1:])


def window_features(xy, conf, boxes):
    """Describe one window with numbers. xy: (T, 17, 2) pixels, conf: (T, 17), boxes: (T, 4)."""
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    size = np.median(height)  # person size, so movements are comparable near and far from the camera
    visible = conf >= MIN_CONF
    points = np.where(visible[:, :, None], xy, np.nan)
    # Pose shape: each keypoint relative to the person's box (0 = left/top edge, 1 = right/bottom edge)
    x_rel = (points[:, :, 0] - boxes[:, [0]]) / width[:, None]
    y_rel = (points[:, :, 1] - boxes[:, [1]]) / height[:, None]

    f = {}
    for i, name in enumerate(NAMES):
        f[f"{name}_x_mean"] = stat(np.mean, x_rel[:, i])
        f[f"{name}_y_mean"] = stat(np.mean, y_rel[:, i])
        f[f"{name}_x_std"] = stat(np.std, x_rel[:, i])
        f[f"{name}_y_std"] = stat(np.std, y_rel[:, i])
        f[f"{name}_visible"] = visible[:, i].mean()

    # Hands and elbows: how fast they move between frames, relative to body size
    for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST):
        step = np.linalg.norm(np.diff(points[:, i], axis=0), axis=1) / size
        f[f"{NAMES[i]}_speed"] = stat(np.mean, step)
        f[f"{NAMES[i]}_speed_max"] = stat(np.max, step)

    # Reaching: how far each hand is from its shoulder, and how often it is above the shoulder
    for wrist, shoulder, side in ((L_WRIST, L_SHOULDER, "l"), (R_WRIST, R_SHOULDER, "r")):
        reach = np.linalg.norm(points[:, wrist] - points[:, shoulder], axis=1) / size
        valid = np.isfinite(reach)
        above = (points[:, wrist, 1] < points[:, shoulder, 1])[valid]
        f[f"{side}_reach_mean"] = stat(np.mean, reach)
        f[f"{side}_reach_max"] = stat(np.max, reach)
        f[f"{side}_hand_above_shoulder"] = above.mean() if above.size else np.nan

    # Head turn: which ear is more visible tells roughly where the head points
    f["ear_balance"] = np.mean(conf[:, L_EAR] - conf[:, R_EAR])

    # Whole body: box shape and how much the person moves in the store
    center = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    f["box_aspect_mean"] = np.mean(width / height)
    f["box_aspect_std"] = np.std(width / height)
    f["body_speed"] = np.mean(np.linalg.norm(np.diff(center, axis=0), axis=1)) / size
    f["body_displacement"] = np.linalg.norm(center[-1] - center[0]) / size
    f["box_height_change"] = height[-1] / height[0]
    return f


def class_windows(table, window, stride):
    """Build all windows for one class table (only 'ok' rows, sorted by frame)."""
    frames = table["frame"].to_numpy()
    splits = table["split"].to_numpy()
    boxes = table[BOX_COLUMNS].to_numpy(dtype=float)
    xy = table[[f"kp{i}_{axis}" for i in range(17) for axis in ("x", "y")]].to_numpy(dtype=float).reshape(-1, 17, 2)
    conf = table[[f"kp{i}_conf" for i in range(17)]].to_numpy(dtype=float)
    rows = []
    for seg_start, seg_end in segments(frames, splits, boxes):
        for start in range(seg_start, seg_end - window + 1, stride):
            end = start + window
            features = window_features(xy[start:end], conf[start:end], boxes[start:end])
            rows.append({"class": table["class"].iloc[0], "split": splits[start],
                         "start_frame": frames[start], "end_frame": frames[end - 1], **features})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keypoints", type=Path, default=Path("data/carr/keypoints"))
    parser.add_argument("--out", type=Path, default=Path("data/carr/windows.csv"))
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.keypoints.glob("*.csv")):
        table = pd.read_csv(path)
        table = table[table["status"] == "ok"].sort_values("frame")
        class_rows = class_windows(table, args.window, args.stride)
        counts = pd.Series([r["split"] for r in class_rows]).value_counts()
        print(f"{path.stem}: {len(class_rows)} windows "
              f"(train {counts.get('train', 0)}, val {counts.get('val', 0)}, test {counts.get('test', 0)})")
        rows += class_rows

    windows = pd.DataFrame(rows)
    windows.to_csv(args.out, index=False)
    print(f"Wrote {len(windows)} windows x {windows.shape[1] - 4} features to {args.out}")


if __name__ == "__main__":
    main()
