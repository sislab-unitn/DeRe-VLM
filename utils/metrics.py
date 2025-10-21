import json
import os
import re
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.svm import SVC


def aggregate_results_per_ent_type(results: dict) -> Dict[str, Dict[str, List[int]]]:
    def _record_results(res: Dict[str, int], results_per_type: dict, ent_type: str):
        if ent_type not in results_per_type:
            results_per_type[ent_type] = {
                "y_true": [],
                "y_pred": [],
            }
        pred, target = res["pred"], res["target"]
        results_per_type[ent_type]["y_true"].append(target)
        results_per_type[ent_type]["y_pred"].append(pred)

    results_per_type: Dict[str, Dict[str, List[int]]] = {}
    for img_id, entities in results.items():
        for ent_type, question_types in entities.items():
            for q_type, q_res in question_types.items():
                if "pred" not in q_res:
                    for sub_q_type, _ in q_res.items():
                        res = results[img_id][ent_type][q_type][sub_q_type]
                        _record_results(res, results_per_type, ent_type)
                else:
                    res = results[img_id][ent_type][q_type]
                    _record_results(res, results_per_type, ent_type)

    return results_per_type


def create_classification_report(
    y_true: List[int],
    y_pred: List[int],
    results_dir: str,
    filename: str = "classification_report.json",
    labels: Optional[List[int]] = None,
    verbose: bool = True,
    save_results: bool = True,
) -> dict:
    if not labels:
        labels = sorted(set(y_true))

    if verbose:
        print(classification_report(y_true, y_pred, zero_division=0, labels=labels))

    report: dict = classification_report(  # type: ignore
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
        labels=labels,
    )
    if "accuracy" not in report:
        report["accuracy"] = accuracy_score(y_true, y_pred)

    if save_results:
        save_path = os.path.join(results_dir, filename)
        os.makedirs(results_dir, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)

    return report


def create_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    results_dir: str,
    filename: str = "confusion_matrix.png",
):
    labels = sorted(set(y_true + y_pred))

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )  # Adjust the figsize to make the plot bigger
    cm = confusion_matrix(y_true, y_pred, normalize="true", labels=labels) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format=".0f")
    plt.tight_layout()

    save_path = os.path.join(results_dir, filename)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def match_number_in_text(text: str) -> int:
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
    }
    digit = re.search(r"\d+", text)
    number = re.search(
        r"(none|zero|one|two|three|four|five|six|seven|eight|nine|ten)",
        text.lower(),
    )
    # take the first digit
    if digit is not None and len(digit.group()) > 0:
        return int(digit.group())
    elif number is not None and len(number.group()) > 0:
        return int(MAP_NUMBERS_TO_DIGITS[number.group()])
    else:
        return -1


def compute_two_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    n_folds: int = 3,
) -> np.ndarray:
    if not labels:
        labels = sorted(np.unique(y_true))
    n_labels = len(labels)

    # Initialize a matrix to store the mean accuracy scores
    accuracy_matrix = np.zeros((n_labels, n_labels))

    random.seed(42)
    np.random.seed(42)

    for idx, i in enumerate(labels[:-1]):
        for j in labels[idx + 1 :]:
            # Create a mask for the current pair of classes
            mask = (y_true == i) | (y_true == j)
            X = y_pred[mask]
            y = y_true[mask]

            n_samples = X.shape[0] // 2
            if len(y_true.shape) == 2:
                n_ents = y_true.shape[0]
                n_samples = n_samples // n_ents
                X = X.reshape(n_ents, -1, y_pred.shape[-1])
                y = y.reshape(n_ents, -1)

            samples_id = np.arange(n_samples)
            np.random.shuffle(samples_id)
            ids_per_fold = np.array_split(samples_id, n_folds)

            # Perform cross-validation
            scores = np.zeros(n_folds)
            for n, test_ids in enumerate(ids_per_fold):
                test_ids = np.concatenate([test_ids, test_ids + n_samples])

                # Create test and train sets
                test_mask = np.zeros(n_samples * 2, dtype=bool)
                test_mask[test_ids] = True

                if len(y_true.shape) == 2:
                    test_mask = np.broadcast_to(
                        test_mask, (y_true.shape[0], test_mask.shape[0])
                    )

                X_test = X[test_mask]
                y_test = y[test_mask]

                X_train = X[~test_mask]
                y_train = y[~test_mask]

                clf = SVC(kernel="linear", C=1, random_state=42)
                clf.fit(X_train, y_train)

                score = clf.score(X_test, y_test)
                scores[n] = score

            accuracy_matrix[i - 1, j - 1] = scores.mean()

    return accuracy_matrix


def plot_two_class_accuracy_matrix(
    accuracy_matrix: np.ndarray,
    out_dir: str,
    labels: list,
    filename: str = "two_class_accuracy.png",
):
    # Plot the accuracy matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(accuracy_matrix, interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Accuracy")
    plt.title("Accuracy Matrix for Class i vs. Class j")
    plt.xlabel("Class j")
    plt.ylabel("Class i")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    # Add percentages to each square
    for i in range(len(labels)):
        for j in range(len(labels)):
            if accuracy_matrix[i, j] > 0:  # Only display non-zero values
                plt.text(
                    j,
                    i,
                    f"{accuracy_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if accuracy_matrix[i, j] < accuracy_matrix.max() / 2
                        else "black"
                    ),
                )
    plt.tight_layout(pad=0.5)  # Adjust margins manually
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{filename}", dpi=300)
    plt.close()


def compute_ova_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    n_folds: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    if not labels:
        labels = sorted(np.unique(y_true))
    n_labels = len(labels)

    random.seed(42)
    np.random.seed(42)

    # Create a mask for the current pair of classes
    X = y_pred
    y = y_true

    if len(y.shape) == 1:
        n_samples = X.shape[0] // n_labels
    if len(y.shape) == 2:
        n_samples = X.shape[1] // n_labels

    samples_id = np.arange(n_samples)
    np.random.shuffle(samples_id)
    ids_per_fold = np.array_split(samples_id, n_folds)

    # Perform cross-validation
    scores = np.zeros(n_folds)
    confusion_matrices = np.zeros((n_folds, n_labels, n_labels))
    for n, test_ids in enumerate(ids_per_fold):
        test_ids = np.concatenate([test_ids + n_samples * i for i in range(n_labels)])

        # Create test and train sets
        test_mask = np.zeros(n_samples * n_labels, dtype=bool)
        test_mask[test_ids] = True

        if len(y_true.shape) == 2:
            test_mask = np.broadcast_to(test_mask, (y.shape[0], test_mask.shape[0]))

        X_test = X[test_mask]
        y_test = y[test_mask]

        X_train = X[~test_mask]
        y_train = y[~test_mask]

        clf = SVC(kernel="linear", C=1, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        c_matrix = confusion_matrix(y_test, preds, labels=labels)
        confusion_matrices[n] = c_matrix

        score = clf.score(X_test, y_test)
        scores[n] = score

    return confusion_matrices, scores


def plot_confusion_matrix(
    confusion_matrices: np.ndarray,
    out_dir: str,
    labels: list,
    filename: str = "OVA_confusion_matrix.png",
):
    confusion_matrix_normalized = confusion_matrices.mean(
        axis=0
    ) / confusion_matrices.mean(axis=0).sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_normalized,
        display_labels=labels,
    )
    disp.plot(cmap="viridis", colorbar=False)

    # Update text annotations to show numbers with one decimal place
    for text in disp.text_.ravel():  # type: ignore
        text.set_text(f"{float(text.get_text())*100:.0f}")

    plt.tight_layout(pad=0.5)  # Adjust margins manually
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{filename}", dpi=300)
    plt.close()
