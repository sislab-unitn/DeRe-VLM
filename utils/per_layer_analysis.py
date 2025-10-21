import argparse
import json
import pickle
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm import tqdm

from metrics import match_number_in_text


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python utils/per_layer_analysis.py",
        description="Compute OVA scores for per-layer representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "pred_folder",
        metavar="PRED_FOLDER",
        type=str,
        help="Folder containing a predictions.pkl file.",
    )
    parser.add_argument(
        "--n-folds",
        metavar="N_FOLDS",
        type=int,
        default=3,
        help="Number of folds for OVA evaluation.",
    )
    parser.add_argument(
        "--n-processes",
        metavar="N_PROCESSES",
        type=int,
        default=4,
        help="Number of processes to use for parallel processing.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def read_predictions_from_pickle(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)  # This will return each dict
            except EOFError:
                break


def compute_ova_scores(y_true, y_pred, n_folds=3):
    np.random.seed(0)

    labels = np.unique(y_true)
    n_labels = len(labels)

    n_samples_per_class = y_true.shape[1] // n_labels
    sample_ids = np.arange(n_samples_per_class)
    sample_ids = np.tile(sample_ids, (y_true.shape[0], 1))
    for row in range(sample_ids.shape[0]):
        np.random.shuffle(sample_ids[row])

    ids_per_fold = np.split(sample_ids, n_folds, axis=1)

    scores = np.zeros(n_folds)
    for fold_id, ids in enumerate(ids_per_fold):

        test_ids = np.concatenate(
            [ids + (n_samples_per_class * i) for i in range(n_labels)], axis=1
        )
        row_indices = np.arange(test_ids.shape[0])[:, np.newaxis]  # Shape (2, 1)

        test_mask = np.zeros_like(y_true, dtype=bool)
        test_mask[row_indices, test_ids] = True

        X_test = y_pred[test_mask]
        y_test = y_true[test_mask]

        X_train = y_pred[~test_mask]
        y_train = y_true[~test_mask]

        clf = SVC(kernel="linear", C=1, random_state=42)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        scores[fold_id] = score

    return scores


def get_method_name(pred_type: str) -> str:
    """Get the name of the method based on the prediction type."""
    if pred_type == "enc_image_features":
        method = "SVM_mean_encoder"
    elif "layer_" in pred_type:
        layer_num = pred_type.split("layer_")[-1]

        if "dec_hidden_states_layer" in pred_type:
            method = "SVM_hs_mean_layer_"
        elif "dec_last_token_layer" in pred_type:
            method = "SVM_hs_last_token_layer_"
        elif "pred_layer" in pred_type:
            method = "LLM_layer_"
        elif "dec_image_features_layer" in pred_type:
            method = "SVM_img_mean_layer_"
        elif "dec_image_last_token_layer" in pred_type:
            method = "SVM_img_last_token_layer_"
        else:
            raise ValueError(f"Invalid prediction type: {pred_type}")

        method += layer_num
    else:
        raise ValueError(f"Invalid prediction type: {pred_type}")

    return method


def get_mean_and_std_for_pred_type(
    results_per_pred_type, y_true
) -> Tuple[float, float, str]:
    pred_type, samples = results_per_pred_type

    y_pred = np.stack([v for v in samples.values()], axis=0)
    method_name = get_method_name(pred_type)

    if "pred" in pred_type:
        y_pred = np.array(list(map(match_number_in_text, y_pred.flatten())))
        acc = accuracy_score(y_pred.flatten(), y_true.flatten())
        std = 0.0
    else:
        scores = compute_ova_scores(y_true, y_pred)
        acc, std = scores.mean(), scores.std()

    return acc, std, method_name  # type: ignore[return-value]


def main(args):
    pred_folder = Path(args.pred_folder)
    assert pred_folder.is_dir(), f"Invalid folder: {pred_folder}"

    predictions = read_predictions_from_pickle(pred_folder / "predictions.pkl")
    results_per_pred_type = {}
    for pred in predictions:
        sample_id = pred["sample_id"]
        _, ent_type, *_ = sample_id.split("-")
        for pred_type in pred:
            if pred_type != "sample_id":
                if pred_type not in results_per_pred_type:
                    results_per_pred_type[pred_type] = {}
                if ent_type not in results_per_pred_type[pred_type]:
                    results_per_pred_type[pred_type][ent_type] = []

                results_per_pred_type[pred_type][ent_type].append(pred[pred_type])

    target = [v for v in results_per_pred_type["target"].values()]
    y_true = np.stack(target, axis=0)

    del results_per_pred_type["target"]

    summary = {}
    with ProcessPoolExecutor(max_workers=args.n_processes) as executor:
        futures = {
            executor.submit(get_mean_and_std_for_pred_type, item, y_true): item[0]
            for item in list(results_per_pred_type.items())
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Computing OVA scores"
        ):
            pred_type = futures[future]
            try:
                result = future.result()
                acc, std, method_name = result
                summary[method_name] = {"accuracy": acc, "std": std}
            except Exception as e:
                print(f"[ERROR] Failed processing {pred_type}: {e}")

    # Save the summary to a file
    summary_file = pred_folder / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
