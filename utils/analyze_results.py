import argparse
import gzip
import re
import json
from argparse import Namespace
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

MAP_NUMBERS_TO_DIGITS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
}


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "results_dir",
        metavar="RESULTS_DIR",
        type=str,
        help="Directory containing the results to analyze.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the ground truths.",
    )
    parser.add_argument(
        "--match-last",
        action="store_true",
        help="Match the last number in the text.",
    )

    return parser.parse_args()


def match_number_in_text(text: str, match_last: bool) -> Optional[str]:
    if match_last:
        digit = re.findall(r"\d+", text)
        number = re.findall(
            r"(none|zero|one|two|three|four|five|six|seven|eight|nine|ten)",
            text.lower(),
        )

        # take the last digit
        if len(digit) > 0:
            return digit[-1]
        elif len(number) > 0:
            return MAP_NUMBERS_TO_DIGITS[number[-1]]
    else:
        digit = re.search(r"\d+", text)
        number = re.search(
            r"(none|zero|one|two|three|four|five|six|seven|eight|nine|ten)",
            text.lower(),
        )
        # take the first digit
        if digit is not None and len(digit.group()) > 0:
            return digit.group()
        elif number is not None and len(number.group()) > 0:
            return MAP_NUMBERS_TO_DIGITS[number.group()]


def convert_predictions_to_labels(args: Namespace, results, ground_truths):

    y_trues = {}
    y_preds = {}
    others = {}
    for img_id, q_types in results.items():
        gt = ground_truths[int(img_id)]["n_stars"]
        for q_type, predictions in q_types.items():
            if "pred" not in predictions:
                for sub_q_type, sub_question in predictions.items():
                    key = f"{q_type}-{sub_q_type}"

                    if key not in y_trues:
                        y_trues[key] = []
                        y_preds[key] = []
                        others[key] = []

                    match = match_number_in_text(args.match_last, sub_question["pred"])
                    y_trues[key].append(gt)
                    if match is not None:
                        y_preds[key].append(int(match))
                    else:
                        y_preds[key].append(-1)
                        others[key].append(img_id)
            else:
                key = f"{q_type}"
                if key not in y_trues:
                    y_trues[key] = []
                    y_preds[key] = []
                    others[key] = []

                match = match_number_in_text(predictions["pred"], args.match_last)
                y_trues[key].append(gt)
                if match is not None:
                    y_preds[key].append(int(match))
                else:
                    y_preds[key].append(-1)
                    others[key].append(img_id)

    for key in y_trues:
        n_others = len(others[key])
        n_total = len(y_trues[key])
        print(f"{key}: Others: # {n_others}, {n_others/n_total*100}%")

    return y_trues, y_preds


def create_classification_report(y_true, y_pred, labels=None):
    if not labels:
        labels = range(max(y_true) + 1)

    print(classification_report(y_true, y_pred, zero_division=0, labels=labels))

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=labels,
    )
    return report


def create_confusion_matrix(y_true, y_pred, results_dir, key, labels=None):
    if not labels:
        labels = range(max(y_true) + 1)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )  # Adjust the figsize to make the plot bigger
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".0f")
    plt.tight_layout()
    plt.savefig(results_dir / f"{key}-confusion_matrix.png")


def main(args: Namespace):
    results_dir = Path(args.results_dir)
    assert results_dir.exists(), f"Results directory {results_dir} does not exist."
    if (results_dir / "generation_results.json").exists():
        with open(results_dir / "generation_results.json", "r") as f:
            results = json.load(f)
    elif (results_dir / "classification_results.json").exists():
        with open(results_dir / "classification_results.json", "r") as f:
            results = json.load(f)
    else:
        raise FileNotFoundError(
            f"Neither generation_results.json nor classification_results.json found in {results_dir}"
        )

    with gzip.open(Path(args.data_dir) / "counting_groundtruths.json.gz", "rt") as f:
        ground_truths = json.load(f)

    y_trues, y_preds = convert_predictions_to_labels(args, results, ground_truths)
    for key in y_trues:
        y_true = y_trues[key]
        y_pred = y_preds[key]
        report = create_classification_report(y_true, y_pred)
        create_confusion_matrix(y_true, y_pred, results_dir, key)

        with open(results_dir / f"{key}-classification_report.json", "w") as f:
            json.dump(report, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
