# Gustavo Ojeda -- Predictive Assistance Alerts: replay task recordings through the live system (Level 1 + Level 2 + alerts)
"""Replay recordings made with live_actions.py --record as if they were live, and score the alerts.

The same code as the live program runs on the saved skeletons (no camera needed): Level 1 action
every 0.67 s, Level 2 "needs help" probability from the last 20 s (actions + body movement), alert
logic with the settings saved in the Level 2 JSON. Each task is treated as a NEW customer: the
Level 2 memory is reset when a task starts (in the lab the volunteer stays in view between tasks,
a real customer arrives from zero).

Each segment of the recording is scored:
    needs help = the volunteer did NOT find the product (task lasted 85 s or more)
    found      = found it (shorter task)
    calm       = normal shopping: every segment of a *_tareas_b file, and Tarea 7 (the 7th task)
Calm segments are also cut into 4 "visits" of ~1 minute each (a new customer each time), closer to a
real store where few customers stay long in one aisle.

Reports: needs-help tasks with an alert (recall), found tasks with an alert, calm segments and calm
1-minute visits with an alert, alerts between tasks, seconds from the start of the task to the alert.
Compared with a timer that alerts at --timer seconds (it knows when each task started).

Example (final exam, run ONCE with the frozen system):
    python -m scripts.replay_level2 --level2 data/pilot/level2_v1.json data/pilot/p09_tareas_a.csv data/pilot/p10_tareas_a.csv data/pilot/p09_tareas_b.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.level2 import BehaviorTracker, IdRelinker, Level2Model
from scripts.live_actions import ActionRecognizer

BOX = ["box_x1", "box_y1", "box_x2", "box_y2"]
XY = [f"kp{i}_{a}" for i in range(17) for a in ("x", "y")]
CONF = [f"kp{i}_conf" for i in range(17)]
HELP_SECONDS = 85
CALM_TASK = 7          # Tarea 7 = "look around and take any two products": normal shopping


def level1_events(table, recognizer):
    """Run Level 1 on every frame. Returns a list of (time, person, box, xy, conf, new action or None)."""
    relinker, events = IdRelinker(), []
    t = table.sort_values(["frame", "track_id"])
    times, frames, tids = t["time"].to_numpy(float), t["frame"].to_numpy(int), t["track_id"].to_numpy(int)
    boxes, xys, confs = t[BOX].to_numpy(float), t[XY].to_numpy(float).reshape(-1, 17, 2), t[CONF].to_numpy(float)
    for now, frame, tid, box, xy, conf in zip(times, frames, tids, boxes, xys, confs):
        person = relinker.update(tid, now, box)
        prediction = recognizer.update(person, frame, now, xy, conf, box)
        new = prediction[0] if prediction and recognizer.last_predicted.get(person) == now else None
        events.append((now, person, box, xy, conf, new))
    return events


def level2_pass(events, level2=None, resets=(), collect=False):
    """Run Level 2 over the Level 1 events. At each time in `resets` everything is forgotten (new customer).
    Returns (alerts [(time, person, probability)], rows of features if collect)."""
    resets = sorted(resets); k = 0
    behavior, alerts = BehaviorTracker(), (level2.alert_logic() if level2 else None)
    fired, rows = [], []
    for now, person, box, xy, conf, new in events:
        while k < len(resets) and now >= resets[k]:
            behavior, alerts = BehaviorTracker(), (level2.alert_logic() if level2 else None); k += 1
        behavior.add_frame(person, now, xy, conf, box)
        if new is None:
            continue
        behavior.add(person, now, new, box)
        values = behavior.features(person)
        if not values:
            continue
        age = behavior.age(person, now)
        if collect:
            rows.append({"time": now, "person_id": person, "age": age, **values})
        if level2:
            smoothed, send = alerts.update(person, now, level2.probability(values), age)
            if send:
                fired.append((now, person, smoothed))
    return fired, rows


def tasks_of(table, calm_file=False):
    """Segments from the n/h key labels: card, start, end, duration and kind (needs_help / found / calm)."""
    frames = table.groupby("frame").first().sort_index()
    label, time = frames["task"].to_numpy(), frames["time"].to_numpy()
    out, start = [], None
    for i in range(len(label)):
        if label[i] != "none" and (i == 0 or label[i] != label[i - 1]):
            start = i
        if start is not None and (i == len(label) - 1 or label[i + 1] != label[i]):
            duration = time[i] - time[start]
            number = len(out) + 1
            kind = "calm" if calm_file or number == CALM_TASK else ("needs_help" if duration >= HELP_SECONDS else "found")
            out.append({"task": number, "card": label[i], "start": time[start], "end": time[i],
                        "duration": round(duration, 1), "kind": kind})
            start = None
    return pd.DataFrame(out)


def visits_of(tasks, parts=4):
    """Cut every calm segment into `parts` visits of equal length (each one a new customer)."""
    out = []
    for _, t in tasks[tasks.kind == "calm"].iterrows():
        length = (t.end - t.start) / parts
        out += [{"task": t.task, "start": t.start + i * length, "end": t.start + (i + 1) * length} for i in range(parts)]
    return pd.DataFrame(out, columns=["task", "start", "end"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=Path("data/pilot/level1_v3.joblib"))
    parser.add_argument("--level2", type=Path, default=Path("data/pilot/level2_v1.json"))
    parser.add_argument("--timer", type=float, default=45.0, help="seconds for the timer comparison")
    parser.add_argument("--out", type=Path, help="save the results as JSON")
    args = parser.parse_args()

    level2 = Level2Model(args.level2)
    rows, visit_rows, between = [], [], 0
    for path in args.recordings:
        table = pd.read_csv(path)
        tasks = tasks_of(table, calm_file=path.stem.endswith("_tareas_b"))
        events = level1_events(table, ActionRecognizer(args.model))
        # new customer at the start of each task; the time between tasks (reading the next card) is its own "visit"
        fired, _ = level2_pass(events, level2, resets=tasks.start.tolist() + (tasks.end + 0.01).tolist())
        used = set()
        for _, t in tasks.iterrows():
            inside = [a for a in fired if t.start <= a[0] <= t.end]
            used |= {a[0] for a in inside}
            rows.append({"recording": path.stem, "task": int(t.task), "card": t.card, "duration": t.duration, "kind": t.kind,
                         "alert": bool(inside), "alert_after_s": round(inside[0][0] - t.start, 1) if inside else None,
                         "timer_alert": t.duration > args.timer})
        between += sum(1 for a in fired if a[0] not in used)
        visits = visits_of(tasks)
        if len(visits):
            vfired, _ = level2_pass(events, level2, resets=visits.start.tolist())
            for _, v in visits.iterrows():
                visit_rows.append({"recording": path.stem, "task": int(v.task), "alert": any(v.start <= a[0] < v.end for a in vfired),
                                   "timer_alert": v.end - v.start > args.timer})
        print(f"{path.stem}: {len(fired)} alerts, {len(tasks)} segments, {len(visits)} calm 1-min visits")

    R, V = pd.DataFrame(rows), pd.DataFrame(visit_rows, columns=["recording", "task", "alert", "timer_alert"])
    print("\n" + R.to_string(index=False))

    def summary(col):
        h, f, c = R[R.kind == "needs_help"], R[R.kind == "found"], R[R.kind == "calm"]
        return {"needs_help_detected": f"{int(h[col].sum())}/{len(h)}", "found_with_alert": f"{int(f[col].sum())}/{len(f)}",
                "calm_segments_with_alert": f"{int(c[col].sum())}/{len(c)}",
                "calm_1min_visits_with_alert": f"{int(V[col].sum())}/{len(V)}" if len(V) else "-"}

    behavior = summary("alert"); behavior["alerts_between_tasks"] = between
    delays = R[R.kind == "needs_help"].alert_after_s.dropna()
    behavior["median_seconds_to_alert"] = round(float(delays.median()), 1) if len(delays) else None
    timer = summary("timer_alert")
    print(f"\nBehavior model ({level2.name}): {behavior}")
    print(f"Timer at {args.timer:.0f} s (knows each task start): {timer}")
    if args.out:
        args.out.write_text(json.dumps({"behavior": behavior, "timer": timer, "tasks": rows, "visits": visit_rows}, indent=2))
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
