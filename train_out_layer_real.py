import argparse
import json
import os
import random
from argparse import Namespace
from functools import partial
from typing import List, Tuple

import torch
import numpy as np
import wandb
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange
from transformers import AutoProcessor


from utils.metrics import (
    create_classification_report,
    create_confusion_matrix,
    match_number_in_text,
)
from utils.utils import MAP_MODELS, read_pickle
from utils.training import seed_worker


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m train_out_layer",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        choices=MAP_MODELS.keys(),
        type=str,
        help="Name of the model to use.",
    )
    parser.add_argument(
        "train_folder",
        metavar="TRAIN_FOLDER",
        type=str,
        help="Folder containing train set.",
    )
    parser.add_argument(
        "valid_folder",
        metavar="VALID_FOLDER",
        type=str,
        help="Folder containing validation set.",
    )
    parser.add_argument(
        "test_folder",
        metavar="TEST_FOLDER",
        type=str,
        help="Folder containing test set.",
    )
    parser.add_argument(
        "exp_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="output",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=50,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, only evaluate the model on the test set.",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help="If set, train the output layer from scratch.",
    )
    parser.add_argument(
        "--kl-div",
        action="store_true",
        help="If set, compute the KL divergence between the predicted logits and the target logits.",
    )
    parser.add_argument(
        "--lambda-kl",
        metavar="LAMBDA_KL",
        type=float,
        default=0.2,
        help="Lambda for the KL divergence loss.",
    )

    parsed_args = parser.parse_args()

    return parsed_args


class CustomDataset(Dataset):
    def __init__(self, data_folder, processor, ids=None):
        self.inputs: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.answers: List[str] = []
        self.sample_ids: List[str] = []

        data_generator = read_pickle(os.path.join(data_folder, "representations.pkl"))
        for sample_id, sample in enumerate(data_generator):
            if ids is None or sample_id in ids:
                self.inputs.append(sample["last_token_repr"])
                token_id = processor.tokenizer.convert_tokens_to_ids(
                    str(sample["target"])
                )
                self.labels.append(token_id)
                self.answers.append(sample["target"])
                self.sample_ids.append(sample["sample_id"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str, str]:
        return (
            self.inputs[idx],
            self.labels[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    total_loss: float,
    unmasked_tokens: int,
) -> Tuple[float, float, float]:
    nll = total_loss / unmasked_tokens
    ppl = 2**nll

    accuracy: float = accuracy_score(y_true, y_pred)  # type: ignore

    return nll, ppl, accuracy


def update_stats(
    logits: torch.Tensor,
    processor,
    y_true: List[int],
    y_pred: List[int],
    mask: torch.Tensor,
    total_loss: float,
    unmasked_tokens: int,
    answers: List[int],
    batch_unmasked_tokens: int,
    loss: torch.Tensor,
):

    total_loss += loss.item()
    unmasked_tokens += batch_unmasked_tokens

    pred = logits.argmax(dim=-1)
    pred = pred[mask].reshape(mask.size(0), -1)
    original_pred = processor.tokenizer.batch_decode(pred, skip_special_tokens=True)
    y_pred += list(map(match_number_in_text, original_pred))
    y_true += answers

    return total_loss, unmasked_tokens, y_true, y_pred


def get_loaders(processor, args) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CustomDataset(args.train_folder, processor)
    val_dataset = CustomDataset(args.valid_folder, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_model(
    in_features, out_features, has_bias, weights, train_from_scratch, requires_grad=True
):
    model = torch.nn.Linear(
        in_features,
        out_features,
        bias=has_bias,
        dtype=weights.dtype,
    )
    if not train_from_scratch:
        model.weight = torch.nn.Parameter(weights)

    model.requires_grad_(requires_grad)

    return model


def train_one_epoch(
    model, train_loader, processor, optimizer, criterion, args, target_model=None
) -> Tuple[float, float, float, float]:

    unmasked_tokens = 0
    y_true = []
    y_pred = []
    next_token_pred_loss = 0.0
    kl_div_loss = 0.0

    model.train()
    for batch in train_loader:
        inputs, labels, answers, _ = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()

        logits = model(inputs)
        # compute the loss as sum
        loss = criterion(logits, labels)
        # store the original loss
        next_token_pred_loss += loss.item()
        # compute the number of unmasked tokens
        mask = labels != -100
        batch_unmasked_tokens = mask.sum().item()

        # normalize the loss
        loss /= batch_unmasked_tokens

        # update the predictions for the current batch
        pred = logits.argmax(dim=-1)
        pred = pred[mask].reshape(mask.size(0), -1)
        original_pred = processor.tokenizer.batch_decode(pred, skip_special_tokens=True)
        y_pred += list(map(match_number_in_text, original_pred))
        y_true += answers

        if target_model:
            with torch.no_grad():
                target_logits = target_model(inputs)

            kl_div = compute_kl_div(
                input_logits=logits,
                target_logits=target_logits,
                mask=mask,
                device=args.device,
            )
            # multiply kl_div by lambda
            kl_div *= args.lambda_kl
            # store the original loss
            kl_div_loss += kl_div.item()
            # normalize the kl_div_loss
            kl_div /= batch_unmasked_tokens

            loss += kl_div

        loss.backward()
        optimizer.step()

        unmasked_tokens += batch_unmasked_tokens

    nll, ppl, accuracy = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        total_loss=next_token_pred_loss,
        unmasked_tokens=unmasked_tokens,
    )

    kl_div_loss /= unmasked_tokens

    return nll, ppl, accuracy, kl_div_loss


def compute_kl_div(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    # move the target logits to the same device as the input logits
    target_logits = target_logits.to(device)
    input_logits = input_logits.to(device)
    mask = mask.to(device)

    # compute the KL divergence between the predicted logits and the target logits
    kl_div = F.kl_div(
        F.log_softmax(input_logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)

    assert (
        kl_div.size() == mask.size()
    ), f"KL Div size {kl_div.size()} does not match mask size {mask.size()}"

    # consider only non padded tokens
    kl_div = (kl_div * mask).sum()

    return kl_div


def evaluate(
    model, val_loader, processor, criterion, args, target_model=None
) -> Tuple[float, float, float, float]:

    unmasked_tokens = 0
    y_true = []
    y_pred = []
    next_token_pred_loss = 0.0
    kl_div_loss = 0.0

    model.eval()
    for batch in val_loader:
        inputs, labels, answers, _ = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            logits = model(inputs)

        # compute the loss as sum
        loss = criterion(logits, labels)
        # store the original loss
        next_token_pred_loss += loss.item()
        # compute the number of unmasked tokens
        mask = labels != -100
        batch_unmasked_tokens = mask.sum().item()

        # normalize the loss
        loss /= batch_unmasked_tokens

        # update the predictions for the current batch
        pred = logits.argmax(dim=-1)
        pred = pred[mask].reshape(mask.size(0), -1)
        original_pred = processor.tokenizer.batch_decode(pred, skip_special_tokens=True)
        y_pred += list(map(match_number_in_text, original_pred))
        y_true += answers

        if target_model:
            with torch.no_grad():
                target_logits = target_model(inputs)

            kl_div = compute_kl_div(
                input_logits=logits,
                target_logits=target_logits,
                mask=mask,
                device=args.device,
            )
            # multiply kl_div by lambda
            kl_div *= args.lambda_kl
            # store the original loss
            kl_div_loss += kl_div.item()
            # normalize the kl_div
            kl_div /= batch_unmasked_tokens

            loss += kl_div

        unmasked_tokens += batch_unmasked_tokens

    nll, ppl, accuracy = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        total_loss=next_token_pred_loss,
        unmasked_tokens=unmasked_tokens,
    )

    kl_div_loss /= unmasked_tokens

    return nll, ppl, accuracy, kl_div_loss


def train(
    lr: float,
    in_features: int,
    out_features: int,
    has_bias: bool,
    weights,
    processor,
    args: Namespace,
    save_best: bool = False,
    verbose: bool = False,
):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb_config = {
        "lr": lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "model_name": args.model_name,
    }

    # Start a new wandb run for this trial
    wandb.init(
        project=f"{args.model_name}-{args.exp_name}",
        config=wandb_config,  # log any params
        reinit="finish_previous",
    )

    model = get_model(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        weights=weights,
        train_from_scratch=args.train_from_scratch,
    ).to(args.device)

    target_model = None
    if args.kl_div:
        # If KL divergence is used, we need to create a target model
        target_model = get_model(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            weights=weights,
            train_from_scratch=False,
            requires_grad=False,  # We don't need to train it anymore
        ).to(args.device)
        target_model.eval()

    train_loader, val_loader = get_loaders(processor, args)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # num_training_steps = math.ceil(args.epochs * len(train_loader))
    # scheduler = get_linear_schedule_with_warmup(
    #    optimizer,
    #    num_warmup_steps=int(0.1 * num_training_steps),
    #    num_training_steps=num_training_steps,
    # )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    best_acc = 0.0
    for epoch in trange(args.epochs, desc="Training Epochs", disable=not verbose):
        train_nll, train_ppl, train_acc, train_kl = train_one_epoch(
            model=model,
            train_loader=train_loader,
            processor=processor,
            optimizer=optimizer,
            criterion=criterion,
            args=args,
            target_model=target_model,
        )

        wandb.log(
            {
                "train_nll": train_nll,
                "train_ppl": train_ppl,
                "train_acc": train_acc,
                "train_kl": train_kl,
            },
            step=epoch,
        )

        val_nll, val_ppl, val_acc, val_kl = evaluate(
            model=model,
            val_loader=val_loader,
            processor=processor,
            criterion=criterion,
            args=args,
            target_model=target_model,
        )

        scheduler.step(val_acc)

        wandb.log(
            {
                "val_nll": val_nll,
                "val_ppl": val_ppl,
                "val_acc": val_acc,
                "val_kl": val_kl,
                "lr": scheduler.get_last_lr()[0],
            },
            step=epoch,
        )

        if best_acc < val_acc:
            best_acc = val_acc
            # Save the model
            if save_best:
                model_save_path = os.path.join(args.out_dir, f"best_model.pt")
                torch.save(model.state_dict(), model_save_path)
                wandb.save(model_save_path)

    return best_acc


def test(model, test_loader, processor, args):
    results = {}
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, labels, answers, sample_ids = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            logits = model(inputs)

            mask = labels != -100
            preds = logits.argmax(dim=-1)
            preds = preds[mask].reshape(mask.size(0), -1)

            for sample_id, pred, answer in zip(sample_ids, preds, answers):

                original_pred = processor.decode(pred, skip_special_tokens=True)
                pred = match_number_in_text(original_pred)

                _, q_id = sample_id.split("-")

                results[q_id] = {
                    "target": answer.item(),
                    "original_pred": original_pred,
                    "pred": pred,
                }

                y_true.append(answer.item())
                y_pred.append(pred)

    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
        json.dump(results, f, indent=4)

    create_classification_report(y_true, y_pred, args.out_dir)

    create_confusion_matrix(y_true, y_pred, args.out_dir)


def main(args):
    args.out_dir = os.path.join(args.out_dir, args.model_name, args.exp_name)
    os.makedirs(args.out_dir, exist_ok=True)

    model_name, Model, _ = MAP_MODELS[args.model_name]

    model = Model(
        model_name,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    Processor = partial(AutoProcessor.from_pretrained, model_name, use_fast=True)

    processor = Processor()
    if args.model_name == "paligemma2-10b":
        weights = model.lm_head.weight.clone()
        in_features = model.lm_head.in_features
        out_features = model.lm_head.out_features
        has_bias = model.lm_head.bias is not None

        torch.set_float32_matmul_precision("high")
    elif args.model_name == "qwen2-vl-7b":
        weights = model.lm_head.weight.clone()
        in_features = model.lm_head.in_features
        out_features = model.lm_head.out_features
        has_bias = model.lm_head.bias is not None

        processor = Processor(
            min_pixels=100 * 28 * 28,
            max_pixels=2304 * 28 * 28,  # size of 1344x1344 images
        )
    elif "llava" in args.model_name:
        weights = model.language_model.lm_head.weight.clone()
        in_features = model.language_model.lm_head.in_features
        out_features = model.language_model.lm_head.out_features
        has_bias = model.language_model.lm_head.bias is not None
    elif args.model_name == "internvl3-8b":
        weights = model.lm_head.weight.clone()
        in_features = model.lm_head.in_features
        out_features = model.lm_head.out_features
        has_bias = model.lm_head.bias is not None
    else:
        raise NotImplementedError(
            f"Model {args.model_name} is not supported for training the output layer."
        )

    total_params = sum(p.numel() for p in model.parameters())
    params_to_train = weights.numel()
    print(
        f"Percentage of parameters to train: {params_to_train / total_params * 100:.2f}%"
    )

    if args.eval_only:
        model = get_model(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            weights=weights,
            train_from_scratch=False,  # We don't need to train it
            requires_grad=False,  # We don't need to train it
        ).to(args.device)
    else:
        train(
            lr=args.lr,
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            weights=weights,
            processor=processor,
            args=args,
            save_best=True,  # Save the best model
            verbose=True,  # Enable verbose logging
        )

        best_model = torch.load(
            os.path.join(args.out_dir, "best_model.pt"),
        )
        model = get_model(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            weights=best_model["weight"],
            train_from_scratch=False,  # We don't need to train it anymore
            requires_grad=False,  # We don't need to train it anymore
        ).to(args.device)

    # evaluate the model on the test set
    test_dataset = CustomDataset(args.test_folder, processor)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test(
        model=model,
        test_loader=test_loader,
        processor=processor,
        args=args,
    )


if __name__ == "__main__":
    # get the arguments
    args = get_args()

    main(args)
