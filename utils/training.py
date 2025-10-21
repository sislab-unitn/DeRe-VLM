import gc
import json
import math
import os
import random
import tarfile
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft.peft_model import PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel

from utils.metrics import match_number_in_text
from utils.utils import generate


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 1**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_nll_and_ppl(
    losses: List[float], unmasked_tokens: int
) -> Tuple[float, float]:
    nll = sum(losses) / unmasked_tokens
    ppl = math.exp(nll)
    return nll, ppl


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


def compute_js_div(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    # move the target logits to the same device as the input logits
    input_logits = input_logits.to(device)
    target_logits = target_logits.to(device)
    mask = mask.to(device)

    input_prob = F.softmax(input_logits, dim=-1)
    target_prob = F.softmax(target_logits, dim=-1)

    midpoint = (input_prob + target_prob) / 2
    midpoint = midpoint.log()

    input_log = F.log_softmax(input_logits, dim=-1)
    target_log = F.log_softmax(target_logits, dim=-1)

    # compute the KL divergence between the predicted logits and the target logits
    input_div = F.kl_div(
        input=midpoint,
        target=input_log,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)

    target_div = F.kl_div(
        input=midpoint,
        target=target_log,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)

    js_div = (input_div + target_div) / 2

    assert (
        js_div.size() == mask.size()
    ), f"Jensen-Shannon Div size {js_div.size()} does not match mask size {mask.size()}"

    # consider only non padded tokens
    js_div = (js_div * mask).sum()

    return js_div


def compute_aux_loss(
    pretrained_model: PeftModel,
    input_logits: torch.Tensor,
    mask: torch.Tensor,
    input_ids: torch.Tensor,
    args: Namespace,
) -> torch.Tensor:

    input_ids = input_ids.to(pretrained_model.device)  # type: ignore

    # put in eval and disable adapter
    pretrained_model.eval()
    with torch.no_grad():
        with pretrained_model.disable_adapter():
            target_logits: torch.Tensor = pretrained_model(input_ids).logits

    if args.use_jensen_shannon:
        div = compute_js_div(
            input_logits=input_logits,
            target_logits=target_logits,
            mask=mask,
            device=input_logits.device,
        )
    else:
        div = compute_kl_div(
            input_logits=input_logits,
            target_logits=target_logits,
            mask=mask,
            device=input_logits.device,
        )

    # move the pretrained model back to training mode
    pretrained_model.train()

    return div


def tar_filter(tarinfo):
    """Exclude specific folders such as __pycache__."""
    excluded_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        ".env",
    }  # Add other unwanted folders here
    if any(excluded_dir in tarinfo.name for excluded_dir in excluded_dirs):
        return None  # Exclude this file/folder
    return tarinfo  # Include all others


def create_tar_gz(archive_name, paths):
    """
    Create a .tar.gz archive from multiple files and folders, excluding certain directories.

    :param archive_name: Name of the output archive file (e.g., 'backup.tar.gz')
    :param paths: List of file and folder paths to include in the archive
    """
    with tarfile.open(archive_name, "w:gz") as tar:
        for path in paths:
            arcname = os.path.basename(path)  # Store without absolute paths
            tar.add(path, arcname=arcname, filter=tar_filter)


def save_training_params(args: Namespace, output_folder: str) -> None:
    with open(os.path.join(output_folder, "training_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    paths = ["utils", "subparsers", "main.py"]
    create_tar_gz(os.path.join(output_folder, "code.tar.gz"), paths)


def load_training_params(output_folder: str) -> Namespace:
    with open(os.path.join(output_folder, "training_params.json"), "r") as f:
        return Namespace(**json.load(f))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_acc = float("-inf")
        self.stopped = False

    def should_stop(self, current_acc: float) -> bool:
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.counter = 0
        else:
            self.counter += 1

        # save the counter condition for the checkpointer
        self.stopped = self.counter >= self.patience
        return self.stopped


class Checkpoint:
    def __init__(self, args: Namespace):
        self.epoch = 0
        self.step = 0
        self.optimizer: Optional[dict] = None
        self.early_stopping = EarlyStopping(args.max_patience)
        self.train_stats = []
        self.losses = []
        self.unmasked_tokens = 0
        self.kl_div_losses = []
        self.total_time: timedelta = timedelta(0)
        self.best_epoch: Optional[int] = None
        self.y_true: List[int] = []
        self.y_pred: List[int] = []


class Checkpointer:
    def __init__(self, args: Namespace):
        self.checkpoint = Checkpoint(args)

    def update_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        step: int,
        losses: List[float],
        unmasked_tokens: int,
        output_folder: str,
        total_time: timedelta,
        y_true: List[int],
        y_pred: List[int],
        train_stats: Optional[List[dict]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch: Optional[int] = None,
        kl_div_losses: Optional[List[float]] = None,
        best_epoch: Optional[int] = None,
    ):
        os.makedirs(os.path.join(output_folder, "checkpoint"), exist_ok=True)
        torch.save(
            model.language_model.lm_head.state_dict(),  # type: ignore
            os.path.join(output_folder, "checkpoint", f"model.pth"),
        )
        self.checkpoint.optimizer = optimizer.state_dict()
        if epoch is not None:
            self.checkpoint.epoch = epoch
        self.checkpoint.step = step
        self.checkpoint.losses = losses
        self.checkpoint.unmasked_tokens = unmasked_tokens
        self.checkpoint.total_time = total_time
        self.checkpoint.y_true = y_true
        self.checkpoint.y_pred = y_pred

        if train_stats is not None:
            self.checkpoint.train_stats = train_stats
        if early_stopping is not None:
            self.checkpoint.early_stopping = early_stopping
        if kl_div_losses is not None:
            self.checkpoint.kl_div_losses = kl_div_losses
        if best_epoch is not None:
            self.checkpoint.best_epoch = best_epoch

        torch.save(
            self.checkpoint, os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )

    def load_checkpoint(
        self, model: PreTrainedModel, output_folder: str
    ) -> Tuple[Checkpoint, PreTrainedModel]:

        best_model_path = os.path.join(output_folder, "checkpoint", f"model.pth")
        model.language_model.lm_head.load_state_dict(torch.load(best_model_path))  # type: ignore
        self.checkpoint = torch.load(
            os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
            weights_only=False,  # torch 2.6 behavior
        )
        return self.checkpoint, model  # type: ignore


def resume_training(epoch: int, args: Namespace, early_stopping: EarlyStopping) -> bool:
    if epoch >= args.epochs:
        print(f"Reached maximum number of epochs {epoch}.")
        return False
    if early_stopping.stopped:
        print(f"Early stopping at epoch {epoch}")
        return False
    return True


def train_one_epoch(
    args,
    model: PreTrainedModel,
    processor,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_iterator: Iterator,
    dataloader: DataLoader,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
) -> Tuple[Tuple[float, float], float, int, timedelta]:

    model.train()
    losses = checkpointer.checkpoint.losses
    unmasked_tokens = checkpointer.checkpoint.unmasked_tokens
    total_time = checkpointer.checkpoint.total_time

    # Resume training for the current step
    for _ in range(start_step):
        next(train_iterator)

    y_true = checkpointer.checkpoint.y_true
    y_pred = checkpointer.checkpoint.y_pred
    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, _, answers, *_) in enumerate(
            train_iterator, start=start_step
        ):
            start = datetime.now()

            input_ids = input_ids.to(args.device)

            optimizer.zero_grad()
            out = model(**input_ids)

            losses.append(out.loss.item())

            out.loss.backward()
            optimizer.step()

            mask = input_ids.labels[:, 1:] != -100
            pred = out.logits[:, :-1, :].argmax(dim=-1)
            pred = pred[mask].reshape(mask.size(0), -1)
            original_pred = processor.tokenizer.batch_decode(
                pred, skip_special_tokens=True
            )

            y_pred += list(map(match_number_in_text, original_pred))
            y_true += answers
            print(original_pred, answers)

            unmasked_tokens += (input_ids.labels[:, 1:] != -100).sum().item()
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)

            postfix = {
                "Train Accuracy": accuracy_score(y_true, y_pred),
                "Train PPL": ppl,
            }

            pbar.set_postfix(postfix)
            pbar.update(1)
            steps_so_far += 1

            end = datetime.now()
            batch_time = end - start
            total_time += batch_time

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,  # Increment the current step
                    losses=losses,
                    unmasked_tokens=unmasked_tokens,
                    output_folder=output_folder,
                    total_time=total_time,
                    y_true=y_true,
                    y_pred=y_pred,
                )

    accuracy: float = accuracy_score(y_true, y_pred)  # type: ignore

    return (
        compute_nll_and_ppl(losses, unmasked_tokens),
        accuracy,
        steps_so_far,
        total_time,
    )


def evaluate(
    args,
    model: PreTrainedModel,
    processor,
    dataloader: DataLoader,
    is_test: bool = False,
) -> Tuple[Tuple[float, float], float]:

    losses = []
    y_true = []
    y_pred = []
    model.eval()  #  type: ignore
    unmasked_tokens = 0
    with torch.no_grad():
        for input_ids, _, answers, *_ in (pbar := tqdm(dataloader, desc="Evaluating")):
            input_ids = input_ids.to(args.device)

            with torch.no_grad():
                out = model(**input_ids)
            losses.append(out.loss.item())
            unmasked_tokens += (input_ids.labels[:, 1:] != -100).sum().item()
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)

            mask = input_ids.labels[:, 1:] != -100
            pred = out.logits[:, :-1, :].argmax(dim=-1)
            pred = pred[mask].reshape(mask.size(0), -1)
            original_pred = processor.tokenizer.batch_decode(
                pred, skip_special_tokens=True
            )

            y_pred += list(map(match_number_in_text, original_pred))
            y_true += answers

            prefix = "Test" if is_test else "Valid"

            pbar.set_postfix(
                {
                    f"{prefix} Accuracy": accuracy_score(y_true, y_pred),
                    f"{prefix} PPL": ppl,
                }
            )
    accuracy: float = accuracy_score(y_true, y_pred)  # type: ignore
    return compute_nll_and_ppl(losses, unmasked_tokens), accuracy


def train(
    args: Namespace,
    model: PreTrainedModel,
    processor,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,  # type: ignore
    output_folder: str,
    checkpointer: Checkpointer,
) -> None:

    start_epoch = checkpointer.checkpoint.epoch
    start_step = checkpointer.checkpoint.step
    train_stats = checkpointer.checkpoint.train_stats
    early_stopping: EarlyStopping = checkpointer.checkpoint.early_stopping
    best_epoch = checkpointer.checkpoint.best_epoch

    steps_so_far = len(train_loader) * start_epoch + start_step

    # Resume training for the current epoch
    for _ in range(start_epoch):
        iter(train_loader)

    for epoch in trange(
        start_epoch,
        args.epochs,
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
    ):
        train_iterator = iter(train_loader)
        (train_nnl, train_ppl), train_acc, steps_so_far, total_time = train_one_epoch(
            args=args,
            model=model,
            processor=processor,
            optimizer=optimizer,
            train_iterator=train_iterator,
            dataloader=train_loader,
            start_step=start_step,
            steps_so_far=steps_so_far,
            checkpointer=checkpointer,
            output_folder=output_folder,
        )

        gc.collect()
        torch.cuda.empty_cache()

        (valid_nnl, valid_ppl), valid_acc = evaluate(
            args=args,
            model=model,
            processor=processor,
            dataloader=valid_loader,
            is_test=False,  # Set to True if you want to evaluate on test set
        )

        gc.collect()
        torch.cuda.empty_cache()

        if valid_acc > early_stopping.best_acc:
            torch.save(
                model.language_model.lm_head.state_dict(),  # type: ignore
                os.path.join(output_folder, "best_model.pth"),
            )
            best_epoch = epoch + 1

        # early stopping also updates the best accuracy
        should_stop = early_stopping.should_stop(valid_acc)

        train_stats.append(
            {
                "Epoch": epoch + 1,
                "Best Epoch": best_epoch,  # Update it
                "Time": str(total_time),
                "Train NLL": train_nnl,
                "Train PPL": train_ppl,
                "Valid NLL": valid_nnl,
                "Valid PPL": valid_ppl,
                "Patience": early_stopping.patience - early_stopping.counter,
                "Train Accuracy": train_acc,
                "Valid Accuracy": valid_acc,
            }
        )

        # Update the checkpointer
        checkpointer.update_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,  # Reset the step
            losses=[],  # Reset the losses
            unmasked_tokens=0,  # Reset the unmasked tokens
            output_folder=output_folder,
            total_time=timedelta(0),  # Reset the total time
            y_true=[],  # reset the y_true
            y_pred=[],  # reset the y_pred
            train_stats=train_stats,
            early_stopping=early_stopping,
            epoch=epoch + 1,  # Increment the epoch
            kl_div_losses=[],  # Reset the kl_div_losses
            best_epoch=best_epoch,
        )

        start_step = 0

        with open(os.path.join(output_folder, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=4)

        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break


def test(
    args,
    model: PreTrainedModel,
    processor,
    dataloader: DataLoader,
):
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for input_ids, _, answers, *_ in tqdm(dataloader, desc="Testing"):
            generated_answers = generate(
                input_ids=input_ids,
                model=model,
                model_name=args.model_name,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
            )
