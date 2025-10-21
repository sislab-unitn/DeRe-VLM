import json
from pathlib import Path
from sklearn.metrics import f1_score

from metrics import create_confusion_matrix

results_path = Path("output/paligemma2-10b/baseline/pixmo-count/results.json")

with open("data/pixmo-count/test/samples.json", "r") as f:
    samples = json.load(f)

with open(results_path, "r") as f:
    results = json.load(f)

pred_per_label = {}
for i, sample in enumerate(samples):
    prediction = results[str(i)]

    label = sample["label"]
    if label not in pred_per_label:
        pred_per_label[label] = {
            "y_true": [],
            "y_pred": [],
        }
    pred_per_label[label]["y_true"].append(prediction["target"])
    pred_per_label[label]["y_pred"].append(prediction["pred"])

summary = {}
for label, preds in pred_per_label.items():
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    f1 = f1_score(y_true, y_pred, average="macro")
    summary[label] = {
        "f1_score": f1,
        "# samples": len(y_true),
        "# classes": len(set(y_true)),
    }
    label_dir = results_path.parent / label
    label_dir.mkdir(parents=True, exist_ok=True)
    if len(set(y_true)) > 1:
        create_confusion_matrix(y_true, y_pred, label_dir, f"confusion_matrix.png")


summary = dict(
    sorted(summary.items(), key=lambda item: item[1]["f1_score"], reverse=True)
)

with open(results_path.parent / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)
