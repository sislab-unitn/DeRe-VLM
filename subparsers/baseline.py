import argparse
import json
import os
import random
from functools import partial

import open_clip
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CLIPProcessor


from utils import coca, clip, qwen2_vl, llava_next, paligemma
from utils.data import CountBenchDataset, PixmoCountDataset
from utils.utils import ENCODER_MODELS, MAP_MODELS, classify, generate
from utils.metrics import (
    create_classification_report,
    create_confusion_matrix,
    match_number_in_text,
)


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "baseline",
        help="Compute baseline performance on CountBenchQA or Pixmo-Count.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--instruction",
        metavar="INSTR",
        type=str,
        default="Answer using as few words as possible",
        help="Instruction used for generation (encoder-decoder only).",
    )
    parser.add_argument(
        "--max-new-tokens",
        metavar="N_TOKENS",
        type=int,
        default=5,
        help="Maximum number of new tokens to generate (encoder-decoder only).",
    )
    parser.add_argument(
        "--test-on",
        metavar="TEST_ON",
        type=str,
        choices=["CountBenchQA", "pixmo-count"],
        default="CountBenchQA",
        help="Dataset to test on.",
    )
    parser.add_argument(
        "--data-folder",
        metavar="DATA_FOLDER",
        type=str,
        default="data/pixmo-count",
        help="Folder containing the modified pixmo-count dataset.",
    )

    parser.set_defaults(experiment_name="baseline")
    parser.set_defaults(func=main)


def main(args):
    # get the model and processor based on the model name
    model_name, Model, Collator = MAP_MODELS[args.model_name]

    if args.model_name == "CLIP":
        model = Model(model_name, device_map="auto" if args.parallel else args.device)
        processor = CLIPProcessor.from_pretrained(
            model_name,
            do_resize=True,  # resize shortest edge to 336
            do_center_crop=True,  # take center crop to make it 336x336
            input_data_format="channels_last",  # input images are of shape (H, W, C)
        )
        collator = clip.CountBenchCollator(processor)  # type: ignore
    elif args.model_name == "CoCa":
        model, preprocess = Model(pretrained=model_name, device=args.device)
        tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
        collator = coca.CountBenchCollator(preprocess, tokenizer)
    else:
        model = Model(
            model_name,
            low_cpu_mem_usage=True,
            device_map="auto" if args.parallel else args.device,
        )
        Processor = partial(AutoProcessor.from_pretrained, model_name, use_fast=True)

        if "llava" in args.model_name or args.model_name in [
            "internvl3-8b",
            "paligemma2-10b",
            "qwen2-vl-7b",
        ]:
            if args.model_name == "qwen2-vl-7b":
                processor = Processor(
                    min_pixels=100 * 28 * 28,
                    max_pixels=2304 * 28 * 28,  # size of 1344x1344 images
                )
            else:
                processor = Processor()

            collator = Collator(processor, args.instruction)
            if args.model_name == "paligemma2-10b":
                torch.set_float32_matmul_precision("high")
        else:
            raise ValueError(f"Invalid model name: {args.model_name}")

        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

    # model config
    model.eval()

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_encoder = args.model_name in ENCODER_MODELS

    if args.test_on == "pixmo-count":
        test_ds = PixmoCountDataset(f"{args.data_folder}/test")
        test_loader = DataLoader(
            test_ds,
            batch_size=1 if is_encoder else args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
        )
    elif args.test_on == "CountBenchQA":
        test_ds = CountBenchDataset(is_encoder=is_encoder)
        test_loader = DataLoader(
            test_ds,
            batch_size=1 if is_encoder else args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collator,
        )
    else:
        raise ValueError(f"Invalid test_on value: {args.test_on}")

    args.out_dir = os.path.join(args.out_dir, args.test_on)
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input_ids, input_texts, targets, sample_ids in tqdm(
            test_loader, desc=f"Counting Objects"
        ):
            if is_encoder:
                probs = classify(input_ids, model, args.model_name, args.device)
                _, preds = probs.cpu().max(dim=-1)
            else:
                preds = generate(
                    input_ids, model, args.model_name, processor, args.max_new_tokens
                )

            for i, (sample_id, pred, target) in enumerate(
                zip(sample_ids, preds, targets)
            ):
                _, q_id = sample_id.split("-")

                if is_encoder:
                    input = input_texts[pred.item()]  # type: ignore
                    pred = pred.item() + 2  # type: ignore
                else:
                    input = processor.decode(input_texts.input_ids[i], skip_special_tokens=True)  # type: ignore
                    pred = match_number_in_text(pred)  # type: ignore

                results[q_id] = {
                    "target": target,
                    "input": input,
                    "pred": pred,
                }

                y_true.append(target)
                y_pred.append(pred)

    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
        json.dump(results, f, indent=4)

    create_classification_report(y_true, y_pred, args.out_dir)

    create_confusion_matrix(y_true, y_pred, args.out_dir)
