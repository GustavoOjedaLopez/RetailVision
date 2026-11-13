# Gustavo Ojeda -- Predictive Assistance Alerts: train and validate Level 2 v1 with the SAME code as the live system
"""Train the Level 2 "needs help" model (v1: Level 1 actions + body movement) and validate it.

Why it is built this way:
  - The behavior numbers are computed by replaying the recordings through the live code
    (scripts/replay_level2.py -> level2.BehaviorTracker), so training and live use exactly the same numbers.
  - Level 1 actions for each person come from a Level 1 model trained WITHOUT that person
    (CARR + the other volunteers), so they have the same mistakes as with a new customer.
  - Each task is a new customer (memory reset at the start of the task). Calm shopping sessions
    (*_tareas_b) are also cut into 1-minute visits for the "calm visits with alert" score.
  - Validation is leave-one-person-out: each person is scored by a Level 2 model that never saw them.
  - The threshold is chosen on validation with one fixed rule (best: % needs-help detected minus
    % calm 1-minute visits with an alert). The exam people are NEVER used here.

Example:
    python -m scripts.train_level2 --out data/pilot/level2_v1.2.json --rows data/pilot/level2_rows_v1.2.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.build_carr_windows import GROUPS
from scripts.level2 import AlertLogic
from scripts.live_actions import ActionRecognizer
from scripts.replay_level2 import level1_events, level2_pass, tasks_of, visits_of
from scripts.train_lab_level1 import INFO_COLUMNS, train

FEATURES = ["pct_looking", "pct_touching", "pct_hands_busy", "pct_not engaged",     # path A: Level 1 actions, last 20 s
            "L_aisle_walk", "L_head_scan", "L_hand_high", "L_still"]               # path B: body movement, last 60 s
# v1.1 (10 nov): path B over the last 60 s instead of 20 s. Same validation: same detection of lost shoppers,
# about half the alerts to calm shoppers (1-min visits 35% -> 19%). The 20 s version is kept as level2_v1_20s.json.
ALERT = {"min_age_s": 45.0, "hold_s": 5.0, "smooth": 5, "cooldown_s": 60.0, "max_alerts": 1}
EXAM_PEOPLE = {"p09", "p10", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23"}   # final exam: never used for training or choices
# (11 new volunteers, 10 nov: the first 5 recorded, p11-p15, go to training; p16-p21 are the exam.
#  12 nov: 2 possible extra volunteers, p22-p23, also go to the exam; decided before recording them)
THRESHOLDS = (0.55, 0.60, 0.65)


def collect(path, saved_level1):
    """Behavior rows of one recording: every Level 1 prediction of the main person of each segment."""
    table = pd.read_csv(path)
    tasks = tasks_of(table, calm_file=path.stem.endswith("_tareas_b"))
    events = level1_events(table, ActionRecognizer(saved=saved_level1))
    out = []
    for mode, segments in (("task", tasks), ("visit", visits_of(tasks))):
        _, rows = level2_pass(events, resets=segments.start.tolist(), collect=True)
        rows = pd.DataFrame(rows)
        for i, s in segments.reset_index(drop=True).iterrows():
            inside = rows[(rows.time >= s.start) & (rows.time <= s.end)]
            if inside.empty:
                continue
            kind = s["kind"] if mode == "task" else "calm"
            counts = inside.person_id.value_counts()
            people = counts.index[:1] if kind != "calm" else counts.index[counts >= 30]   # calm: every real person in view
            for pid in people:
                part = inside[inside.person_id == pid].copy()
                part["unit"] = f"{path.stem}_{mode}{i}_{pid}"; part["mode"] = mode; part["kind"] = kind
                part["help"] = int(kind == "needs_help"); part["start"] = s.start
                out.append(part)
    return out


def alerts_for(rows, probs, threshold):
    """Simulate the live alert logic on each unit; return {unit: seconds to the first alert or None}."""
    first = {}
    for unit, g in rows.assign(p=probs).sort_values("time").groupby("unit"):
        logic = AlertLogic(threshold, **{k: v for k, v in ALERT.items()})
        first[unit] = None
        for t, p, age in zip(g.time, g.p, g.age):
            if logic.update(unit, t, p, age)[1]:
                first[unit] = t - g.start.iloc[0]; break
    return first


def score(rows, first):
    units = rows.groupby("unit").agg(mode=("mode", "first"), kind=("kind", "first"))
    got = pd.Series([first[u] is not None for u in units.index], index=units.index)
    pick = lambda m, k: got[(units["mode"] == m) & (units.kind == k)]
    delays = [first[u] for u in units.index[(units["mode"] == "task") & (units.kind == "needs_help")] if first[u] is not None]
    return {"needs_help_detected": [int(pick("task", "needs_help").sum()), int(len(pick("task", "needs_help")))],
            "found_with_alert": [int(pick("task", "found").sum()), int(len(pick("task", "found")))],
            "calm_segments_with_alert": [int(pick("task", "calm").sum()), int(len(pick("task", "calm")))],
            "calm_1min_visits_with_alert": [int(pick("visit", "calm").sum()), int(len(pick("visit", "calm")))],
            "median_seconds_to_alert": round(float(np.median(delays)), 1) if delays else None}


def fit(rows):
    train_rows = rows[(rows["mode"] == "task")].groupby("unit", group_keys=False).apply(lambda g: g.iloc[::15])   # ~one every 10 s
    scaler = StandardScaler().fit(train_rows[FEATURES])
    lr = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(scaler.transform(train_rows[FEATURES]), train_rows.help)
    return scaler, lr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, default=Path("data/pilot"))
    parser.add_argument("--carr-windows", type=Path, default=Path("data/carr/windows.csv"))
    parser.add_argument("--lab-windows", type=Path, default=Path("data/pilot/lab_windows.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/pilot/level2_v1.json"))
    parser.add_argument("--rows", type=Path, help="optional: save/reuse the behavior rows (CSV) to skip the slow replay")
    args = parser.parse_args()

    if args.rows and args.rows.exists():
        rows = pd.read_csv(args.rows)
    else:
        carr = pd.read_csv(args.carr_windows); carr["class"] = carr["class"].map(GROUPS)
        carr = carr[carr.split.isin(["train", "val"])]
        lab = pd.read_csv(args.lab_windows)
        level1_features = [c for c in carr.columns if c not in INFO_COLUMNS]
        recordings = sorted(p for p in args.pilot.glob("p*_tareas_[ab].csv") if p.stem[:3] not in EXAM_PEOPLE)
        people = sorted({p.stem[:3] for p in recordings})
        parts = []
        for person in people:
            # Level 1 that never saw this person (like a new customer)
            saved = {"model": train(carr, lab[lab.person != person], level1_features, 20), "features": level1_features}
            for path in [p for p in recordings if p.stem.startswith(person)]:
                got = collect(path, saved)
                for r in got:
                    r["person"] = person
                parts += got
                print(f"{path.stem}: {len(got)} units", flush=True)
        rows = pd.concat(parts, ignore_index=True)
        if args.rows:
            rows.to_csv(args.rows, index=False)
    assert not set(rows.person) & EXAM_PEOPLE, "exam people must never be used here"

    # leave-one-person-out validation
    probs = np.zeros(len(rows))
    for person in sorted(rows.person.unique()):
        scaler, lr = fit(rows[rows.person != person])
        test = (rows.person == person).values
        probs[test] = lr.predict_proba(scaler.transform(rows.loc[test, FEATURES]))[:, 1]
    results = {}
    for thr in THRESHOLDS:
        results[thr] = score(rows, alerts_for(rows, probs, thr))
        print(f"threshold {thr}: {results[thr]}")
    rule = lambda r: r["needs_help_detected"][0] / max(r["needs_help_detected"][1], 1) - \
        r["calm_1min_visits_with_alert"][0] / max(r["calm_1min_visits_with_alert"][1], 1)
    threshold = max(sorted(THRESHOLDS, reverse=True), key=lambda t: rule(results[t]))   # a tie goes to the higher (safer) threshold
    print(f"chosen threshold (validation rule): {threshold}")

    scaler, lr = fit(rows)
    model = {"name": args.out.stem, "features": FEATURES,   # e.g. level2_v1.2.json -> "level2_v1.2"
             "mean": scaler.mean_.round(6).tolist(), "scale": scaler.scale_.round(6).tolist(),
             "weights": lr.coef_[0].round(6).tolist(), "intercept": round(float(lr.intercept_[0]), 6),
             "threshold": threshold, "alert": ALERT, "segment_seconds": 20,
             "validation": {"method": "leave-one-person-out, each task = new customer", "chosen": results[threshold],
                            "all_thresholds": {str(k): v for k, v in results.items()}},
             "trained_on": sorted(rows.person.unique()), "saved_at": datetime.now().isoformat(timespec="seconds")}
    args.out.write_text(json.dumps(model, indent=2))
    print(json.dumps({k: model[k] for k in ("features", "weights", "intercept", "threshold")}, indent=1))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
