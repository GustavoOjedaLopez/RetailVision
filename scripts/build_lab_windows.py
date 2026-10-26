# Gustavo Ojeda -- Predictive Assistance Alerts: turn our own labeled lab recordings into movement windows
"""Cut the recordings made with live_actions.py --record into windows, like build_carr_windows.py.

Name files with a person code first, e.g. p01_acciones_a.csv, p01_acciones_b.csv, p02_acciones_a.csv:
recordings of the same person are grouped so the exam never mixes them.

A recording has one row per frame and person, with the action label pressed on the keyboard
(1 looking, 2 touching, 3 picking, 4 not engaged, 5 phone; "none" between scenes).

Each window covers the same time as a CARR window (60 frames at 22.5 fps = 2.7 s) and uses
the same features, with speeds converted to CARR's fps. A window gets a label only if ALL its
frames have the same action; windows that touch "none" or mix two actions are dropped, so a
key pressed half a second late does not create wrong examples.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_carr_windows import CARR_FPS, STRIDE, WINDOW, window_features

BOX_COLUMNS = ["box_x1", "box_y1", "box_x2", "box_y2"]
SECONDS = WINDOW / CARR_FPS        # 2.67 s, same as a CARR window
EVERY = STRIDE / CARR_FPS          # a new window every 0.67 s


def person_of(stem):
    """Person code from the file name: 'p03_acciones_b' -> 'p03'. Old names without a code keep their name.

    Gustavo Ojeda -- Predictive Assistance Alerts: all recordings of the same person must share a code,
    so the exam never trains and tests on the same person.
    """
    match = re.match(r"(p\d+)_", stem)
    return match.group(1) if match else stem


def recording_windows(table, session):
    """Windows for one recording. A person's history restarts when their track is interrupted."""
    rows = []
    for track_id, person in table.sort_values("frame").groupby("track_id"):
        frames = person["frame"].to_numpy()
        times = person["time"].to_numpy()
        actions = person["action"].to_numpy()
        boxes = person[BOX_COLUMNS].to_numpy(dtype=float)
        xy = person[[f"kp{i}_{a}" for i in range(17) for a in ("x", "y")]].to_numpy(dtype=float).reshape(-1, 17, 2)
        conf = person[[f"kp{i}_conf" for i in range(17)]].to_numpy(dtype=float)
        breaks = np.flatnonzero(np.diff(frames) != 1) + 1
        for part in np.split(np.arange(len(frames)), breaks):
            next_start = times[part[0]]
            for i in part:
                if times[i] < next_start:
                    continue
                end = np.searchsorted(times, times[i] + SECONDS, side="right")
                if end > part[-1] + 1:
                    break  # not enough recording left in this part for a full window
                next_start = times[i] + EVERY
                if times[end - 1] - times[i] < 0.9 * SECONDS or end - i < 5:
                    continue  # the camera froze for a moment: skip this window, keep going
                labels = set(actions[i:end])
                if len(labels) != 1 or "none" in labels:
                    continue
                fps = (end - i - 1) / (times[end - 1] - times[i])
                features = window_features(xy[i:end], conf[i:end], boxes[i:end], speed_scale=fps / CARR_FPS)
                rows.append({"session": session, "person": person_of(session), "class": actions[i], "track_id": track_id,
                             "start_time": times[i], "end_time": times[end - 1], **features})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", type=Path, nargs="+", help="CSV files from live_actions.py --record")
    parser.add_argument("--out", type=Path, default=Path("data/pilot/lab_windows.csv"))
    args = parser.parse_args()

    rows = []
    for path in args.recordings:
        session_rows = recording_windows(pd.read_csv(path), session=path.stem)
        counts = pd.Series([r["class"] for r in session_rows]).value_counts().to_dict()
        print(f"{path.stem} (person {person_of(path.stem)}): {len(session_rows)} windows {counts}")
        rows += session_rows
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} windows to {args.out}")


if __name__ == "__main__":
    main()
