# Gustavo Ojeda -- Predictive Assistance Alerts: adapt the Level 1 model to our lab camera (CARR + lab data)
"""Train the Level 1 action model with CARR windows plus our own labeled lab recordings.

CARR has thousands of examples but from a ceiling camera in another store; our lab
recordings have few examples but from the camera we will really use. Training on both,
with the lab windows counting more (--lab-weight), gives the best of each.

Evaluation is leave-one-person-out: train with the lab recordings of all people but one, test on
the person left out (the model never saw them). That tells how well it works on a new customer.
People are read from the "person" column (file names like p03_acciones_a.csv).
Scores are reported on the 4 CARR groups and, separately, how often "phone" is recognized
(phone only exists in our lab data).

Finally a model is trained with ALL people and saved, ready for live_actions.py --model.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from scripts.build_carr_windows import GROUPS

INFO_COLUMNS = ["class", "split", "start_frame", "end_frame", "session", "person", "track_id", "start_time", "end_time"]
SEED = 0


def train(carr, lab, features, lab_weight):
    """Gradient Boosting on CARR + lab windows; each lab window counts lab_weight times."""
    x = pd.concat([carr[features], lab[features]])
    y = pd.concat([carr["class"], lab["class"]])
    weights = np.r_[np.ones(len(carr)), np.full(len(lab), lab_weight)]
    return HistGradientBoostingClassifier(random_state=SEED).fit(x, y, sample_weight=weights)


def score(model, test, features, show_details):
    predicted = pd.Series(model.predict(test[features]), index=test.index)
    not_phone = test["class"] != "phone"
    result = {"accuracy_4_groups": accuracy_score(test["class"][not_phone], predicted[not_phone]),
              "phone_recognized": (predicted[~not_phone] == "phone").mean() if (~not_phone).any() else np.nan}
    if show_details:
        labels = sorted(set(test["class"]) | set(predicted))
        matrix = pd.DataFrame(confusion_matrix(test["class"], predicted, labels=labels),
                              index=[f"true: {c}" for c in labels], columns=labels)
        print(matrix.to_string(), "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carr-windows", type=Path, default=Path("data/carr/windows.csv"))
    parser.add_argument("--lab-windows", type=Path, default=Path("data/pilot/lab_windows.csv"))
    parser.add_argument("--lab-weight", type=float, default=20)
    parser.add_argument("--details", action="store_true", help="show confusion matrices")
    parser.add_argument("--save", type=Path, default=Path("data/pilot/level1_v2.joblib"))
    args = parser.parse_args()

    carr = pd.read_csv(args.carr_windows)
    carr["class"] = carr["class"].map(GROUPS)
    carr = carr[carr["split"].isin(["train", "val"])]   # CARR test stays untouched
    lab = pd.read_csv(args.lab_windows)
    features = [c for c in carr.columns if c not in INFO_COLUMNS]
    if "person" not in lab.columns:
        lab["person"] = lab["session"]
    sessions = sorted(lab["person"].unique())
    print(f"CARR windows: {len(carr)} | lab windows: {len(lab)} from people {sessions} | lab weight x{args.lab_weight}\n")

    rows = []
    for test_session in sessions:
        train_lab = lab[lab["person"] != test_session]
        test = lab[lab["person"] == test_session]
        print(f"--- test on {test_session} (train with CARR + {sorted(train_lab['person'].unique())})")
        combined = train(carr, train_lab, features, args.lab_weight)
        rows.append({"test_person": test_session, "model": "CARR + lab", **score(combined, test, features, args.details)})
        baseline = train(carr, train_lab.iloc[0:0], features, args.lab_weight)   # same model without lab data
        rows.append({"test_person": test_session, "model": "CARR only", **score(baseline, test, features, False)})

    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    print("\nAverage over people:")
    print(table.groupby("model")[["accuracy_4_groups", "phone_recognized"]].mean().round(3).to_string())

    final = train(carr, lab, features, args.lab_weight)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": features, "groups": True}, args.save)
    summary = table[table["model"] == "CARR + lab"][["accuracy_4_groups", "phone_recognized"]].mean()
    args.save.with_suffix(".json").write_text(json.dumps({
        "model": "gradient_boosting CARR + lab", "lab_people": sessions, "lab_weight": args.lab_weight,
        "leave_one_person_out_accuracy_4_groups": round(float(summary["accuracy_4_groups"]), 4),
        "leave_one_person_out_phone_recognized": round(float(summary["phone_recognized"]), 4),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"\nSaved final model (all people) to {args.save}")


if __name__ == "__main__":
    main()
