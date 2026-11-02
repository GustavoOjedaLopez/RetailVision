# Gustavo Ojeda -- Predictive Assistance Alerts: final exam of a saved Level 1 model on people it never saw
"""Final exam of the Level 1 model on recordings that were NEVER used for training (p09, p10).

Run it ONCE with the final model. If the result is lower than expected, report it as it is:
changing the model and examining again would turn the exam into one more validation.

Example:
    python -m scripts.exam_level1 --model data/pilot/level1_v3.joblib data/pilot/p09_acciones_a.csv data/pilot/p10_acciones_a.csv

Reports, per person and in total: accuracy on the 4 CARR groups (phone left out), how often
phone is recognized, and the confusion matrix. Saves everything next to the model as *_exam.json.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from scripts.build_lab_windows import person_of, recording_windows

ORDER = ["looking", "touching", "picking", "not engaged", "phone"]


def scores(real, predicted):
    """Accuracy on the 4 groups (without phone) and share of phone windows recognized as phone."""
    not_phone = real != "phone"
    return {"windows": int(len(real)),
            "accuracy_4_groups": round(float(accuracy_score(real[not_phone], predicted[not_phone])), 4),
            "phone_recognized": round(float((predicted[~not_phone] == "phone").mean()), 4) if (~not_phone).any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", type=Path, nargs="+", help="action recordings of people the model never saw")
    parser.add_argument("--model", type=Path, default=Path("data/pilot/level1_v3.joblib"))
    args = parser.parse_args()

    saved = joblib.load(args.model)
    model, features = saved["model"], saved["features"]
    windows = pd.concat([pd.DataFrame(recording_windows(pd.read_csv(path), session=path.stem)) for path in args.recordings])
    windows["predicted"] = model.predict(windows[features])

    results = {"model": str(args.model), "recordings": [p.stem for p in args.recordings],
               "run_at": datetime.now().isoformat(timespec="seconds"), "people": {}}
    print(f"FINAL EXAM of {args.model.name} ({len(windows)} windows)\n")
    for person, part in windows.groupby("person"):
        results["people"][person] = scores(part["class"], part["predicted"])
        print(f"{person}: {results['people'][person]}")
    results["all"] = scores(windows["class"], windows["predicted"])
    results["average_over_people"] = {
        key: round(sum(r[key] for r in results["people"].values()) / len(results["people"]), 4)
        for key in ("accuracy_4_groups", "phone_recognized")}
    print(f"\nAll windows together: {results['all']}")
    print(f"Average over people:  {results['average_over_people']}")

    matrix = pd.crosstab(windows["class"].values, windows["predicted"].values).reindex(index=ORDER, columns=ORDER, fill_value=0)
    matrix.index.name, matrix.columns.name = "real", "predicted"
    percent = (matrix.div(matrix.sum(axis=1), axis=0) * 100).round(0)
    print("\nConfusion matrix, % of each real action (rows = real, columns = predicted):")
    print(percent.to_string())
    results["confusion_counts"] = matrix.to_dict(orient="index")

    out = args.model.with_name(args.model.stem + "_exam.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
