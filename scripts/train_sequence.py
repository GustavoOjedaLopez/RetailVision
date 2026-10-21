# Gustavo Ojeda -- Predictive Assistance Alerts: Level 1 sequence model (1D-CNN) on raw keypoint movement
"""Train a small 1D convolutional network that reads the keypoints frame by frame.

Unlike the Random Forest / Gradient Boosting features (averages over the window),
this model sees the ORDER of the movement: "reach, then pull back" is different
from "pull back, then reach". It uses the same windows, splits and scene-cut rules
as build_carr_windows.py, so the comparison is fair.

Each frame is described by 55 numbers: 17 keypoints x (x, y, confidence) relative
to the person's box, plus how the box moved since the start of the window and its shape.
Training runs on the train split; the epoch with the best validation macro F1 is kept.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from scripts.build_carr_windows import BOX_COLUMNS, GROUPS, MIN_CONF, segments

SEED = 0
# Left/right keypoint pairs, swapped when a window is mirrored (data augmentation)
MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def frame_channels(xy, conf, boxes):
    """(T, 17, 2), (T, 17), (T, 4) -> (T, 55) numbers per frame for one window."""
    center = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    height = boxes[:, 3] - boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    size = np.median(height)
    rel = (xy - center[:, None, :]) / size          # keypoints around the body center, in body heights
    rel[conf < MIN_CONF] = 0.0                      # unreliable points -> 0 (the confidence channel says so)
    moved = (center - center[0]) / size             # how the body moved since the window started
    shape = np.c_[width / height, height / size]
    return np.concatenate([rel.reshape(len(xy), -1), conf, moved, shape], axis=1).astype(np.float32)


def build_sequences(keypoints_dir, window, stride):
    """Return X (N, window, 55), labels, splits and a segment id per window."""
    sequences, labels, splits, segment_ids = [], [], [], []
    segment_counter = 0
    for path in sorted(keypoints_dir.glob("*.csv")):
        table = pd.read_csv(path)
        table = table[table["status"] == "ok"].sort_values("frame")
        frames = table["frame"].to_numpy()
        split = table["split"].to_numpy()
        boxes = table[BOX_COLUMNS].to_numpy(dtype=float)
        xy = table[[f"kp{i}_{a}" for i in range(17) for a in ("x", "y")]].to_numpy(dtype=float).reshape(-1, 17, 2)
        conf = table[[f"kp{i}_conf" for i in range(17)]].to_numpy(dtype=float)
        for seg_start, seg_end in segments(frames, split, boxes):
            segment_counter += 1
            for start in range(seg_start, seg_end - window + 1, stride):
                end = start + window
                sequences.append(frame_channels(xy[start:end], conf[start:end], boxes[start:end]))
                labels.append(path.stem)
                splits.append(split[start])
                segment_ids.append(segment_counter)
    return np.stack(sequences), np.array(labels), np.array(splits), np.array(segment_ids)


def mirror(batch):
    """Flip windows left-right: negate x and swap left/right keypoints. A shopper can reach with either hand."""
    out = batch.clone()
    n = 17
    xy = out[:, :, :2 * n].reshape(len(out), out.shape[1], n, 2)
    xy = xy[:, :, MIRROR]
    xy[..., 0] = -xy[..., 0]
    out[:, :, :2 * n] = xy.reshape(len(out), out.shape[1], 2 * n)
    out[:, :, 2 * n:3 * n] = out[:, :, 2 * n:3 * n][:, :, MIRROR]
    out[:, :, 3 * n] = -out[:, :, 3 * n]  # horizontal body movement
    return out


class ConvNet(nn.Module):
    """Small 1D-CNN: slides filters along time to find movement patterns, then averages over the window."""

    def __init__(self, channels, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128, classes),
        )

    def forward(self, x):            # x: (batch, time, channels)
        return self.net(x.transpose(1, 2))


def predict_proba(model, x, batch_size=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(x[i:i + batch_size]), dim=1)
                          for i in range(0, len(x), batch_size)]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keypoints", type=Path, default=Path("data/carr/keypoints"))
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--groups", action="store_true", help="4 groups instead of 6 actions")
    parser.add_argument("--save", type=Path, default=Path("data/carr/level1_cnn.pt"))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start_time = time.time()
    x, labels, splits, segment_ids = build_sequences(args.keypoints, args.window, args.stride)
    if args.groups:
        labels = np.array([GROUPS[label] for label in labels])
    classes = sorted(set(labels))
    y = np.array([classes.index(label) for label in labels])
    train, val = splits == "train", splits == "val"
    print(f"Windows: train {train.sum()}, val {val.sum()}, test {(splits == 'test').sum()}; "
          f"{x.shape[1]} frames x {x.shape[2]} numbers; classes: {classes} "
          f"({time.time() - start_time:.0f} s to build)")

    # Scale every channel with train statistics only (val/test must not influence training)
    mean = x[train].mean(axis=(0, 1))
    std = x[train].std(axis=(0, 1)) + 1e-6
    x = torch.tensor((x - mean) / std)
    y_t = torch.tensor(y)
    mirror_mean = torch.tensor(mean)  # mirroring must happen in original units, see below

    model = ConvNet(x.shape[2], len(classes))
    weights = torch.tensor(len(y[train]) / (len(classes) * np.bincount(y[train])), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_idx = np.flatnonzero(train)
    std_t, mean_t = torch.tensor(std), mirror_mean

    best = {"f1": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train_idx)
        for i in range(0, len(train_idx), 64):
            batch = train_idx[i:i + 64]
            xb = x[batch]
            flip = torch.rand(len(batch)) < 0.5
            if flip.any():  # undo scaling, mirror, scale again
                raw = xb[flip] * std_t + mean_t
                xb[flip] = (mirror(raw) - mean_t) / std_t
            optimizer.zero_grad()
            loss = loss_fn(model(xb), y_t[batch])
            loss.backward()
            optimizer.step()
        predicted = predict_proba(model, x[val]).argmax(axis=1)
        f1 = f1_score(y[val], predicted, average="macro")
        print(f"epoch {epoch:2d}  loss {loss.item():.3f}  val accuracy {accuracy_score(y[val], predicted):.3f}  "
              f"val macro F1 {f1:.3f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    proba = predict_proba(model, x[val])
    predicted = proba.argmax(axis=1)
    print(f"\nBest epoch {best['epoch']}: val accuracy {accuracy_score(y[val], predicted):.3f}, "
          f"macro F1 {best['f1']:.3f}")
    print(classification_report(y[val], predicted, target_names=classes, digits=3))
    matrix = pd.DataFrame(confusion_matrix(y[val], predicted), index=[f"true: {c}" for c in classes], columns=classes)
    print("Confusion matrix (rows = real action, columns = predicted action):")
    print(matrix.to_string())

    # Smoothing over time: average the predictions of all windows from the same clip
    by_segment = pd.DataFrame(proba).groupby(segment_ids[val]).mean().idxmax(axis=1)
    truth = pd.Series(y[val]).groupby(segment_ids[val]).first()
    print(f"\nClip vote (average all windows of a clip): accuracy {accuracy_score(truth, by_segment):.3f} "
          f"on {len(truth)} clips")

    torch.save({"state": best["state"], "classes": classes, "mean": mean, "std": std,
                "window": args.window}, args.save)
    print(f"Saved model to {args.save}  (total {time.time() - start_time:.0f} s)")


if __name__ == "__main__":
    main()
