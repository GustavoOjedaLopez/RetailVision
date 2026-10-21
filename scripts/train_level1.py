# Gustavo Ojeda -- Predictive Assistance Alerts: Level 1 model, recognize basic shopping actions from movement windows
"""Train and compare models that recognize CARR shopping actions from window features.

Models are trained on the train split and compared on the validation split.
The best one (by validation macro F1) is saved together with its results, so a
version can be kept as a backup and compared against later experiments.
The test split is only used with --test, once, for the final chosen model.

Models compared:
  - majority: always predicts the most common class (the score to beat)
  - random_forest: many decision trees voting (our baseline)
  - gradient_boosting: trees built one after another, each fixing the previous ones' errors
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from scripts.build_carr_windows import GROUPS

INFO_COLUMNS = ["class", "split", "start_frame", "end_frame"]
SEED = 0


def make_models():
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        # Random Forest cannot use missing keypoints directly: fill them with the typical (median) value
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


def evaluate(name, model, x, y, show_details):
    """Print and return the scores of one model on one split."""
    predicted = model.predict(x)
    scores = {"accuracy": accuracy_score(y, predicted), "macro_f1": f1_score(y, predicted, average="macro")}
    print(f"{name:18s} accuracy {scores['accuracy']:.3f}   macro F1 {scores['macro_f1']:.3f}")
    if show_details:
        labels = sorted(y.unique())
        print(classification_report(y, predicted, digits=3))
        matrix = pd.DataFrame(confusion_matrix(y, predicted, labels=labels),
                              index=[f"true: {c}" for c in labels], columns=labels)
        print("Confusion matrix (rows = real action, columns = predicted action):")
        print(matrix.to_string(), "\n")
    return scores


def save(model, name, features, args, scores, split_name):
    """Save the model file and a small JSON with what it is and how well it did."""
    joblib.dump({"model": model, "features": features, "groups": args.groups}, args.save)
    results = {
        "model": name,
        "scored_on": split_name,
        "windows_file": str(args.windows),
        "groups": args.groups,
        "classes": sorted(model.classes_.tolist()),
        "accuracy": round(scores["accuracy"], 4),
        "macro_f1": round(scores["macro_f1"], 4),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    results_path = args.save.with_suffix(".json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {name} to {args.save} and its results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=Path, default=Path("data/carr/windows.csv"))
    parser.add_argument("--groups", action="store_true", help="4 groups instead of 6 actions")
    parser.add_argument("--details", action="store_true", help="per-class report and confusion matrix")
    parser.add_argument("--test", choices=["random_forest", "gradient_boosting"],
                        help="final evaluation on the test split with the chosen model (do this once)")
    parser.add_argument("--save", type=Path, default=Path("data/carr/level1_model.joblib"))
    args = parser.parse_args()

    windows = pd.read_csv(args.windows)
    if args.groups:
        windows["class"] = windows["class"].map(GROUPS)
    features = [c for c in windows.columns if c not in INFO_COLUMNS]
    parts = {name: windows[windows["split"] == name] for name in ("train", "val", "test")}
    print(f"Windows: train {len(parts['train'])}, val {len(parts['val'])}, test {len(parts['test'])}; "
          f"{len(features)} features; classes: {sorted(windows['class'].unique())}\n")

    if args.test:
        # Final step: train the chosen model on train + val, then score it once on test
        model = make_models()[args.test]
        train_val = pd.concat([parts["train"], parts["val"]])
        model.fit(train_val[features], train_val["class"])
        print("FINAL TEST RESULT")
        scores = evaluate(args.test, model, parts["test"][features], parts["test"]["class"], show_details=True)
        save(model, args.test, features, args, scores, "test")
        return

    print("Validation results (train on train, score on val):")
    best = None
    for name, model in make_models().items():
        model.fit(parts["train"][features], parts["train"]["class"])
        scores = evaluate(name, model, parts["val"][features], parts["val"]["class"],
                          show_details=args.details and name != "majority")
        if name == "random_forest" and args.details:
            importance = pd.Series(model[-1].feature_importances_, index=features).sort_values(ascending=False)
            print("Top 15 features the Random Forest relies on:")
            print(importance.head(15).round(4).to_string(), "\n")
        if best is None or scores["macro_f1"] > best[2]["macro_f1"]:
            best = (name, model, scores)

    print(f"\nBest on validation: {best[0]} (macro F1 {best[2]['macro_f1']:.3f})")
    save(best[1], best[0], features, args, best[2], "val")


if __name__ == "__main__":
    main()
