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


from utils import qwen2_vl, llava_next, paligemma
from utils.data import CIVETDataset
from utils.metrics import (
    aggregate_results_per_ent_type,
    create_classification_report,
    create_confusion_matrix,
    match_number_in_text,
)
from utils.utils import ENCODER_MODELS, MAP_MODELS, classify, generate


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the models performance on CIVET.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "data_folder",
        metavar="DATA_FOLDER",
        type=str,
        help="Folder containing train, validation, and test splits.",
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use only 100 questions to check the answer of the model.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save the results after every N steps.",
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
        "--open-ended-questions",
        action="store_true",
        help="Use open ended questions.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override the existing results.",
    )

    parser.set_defaults(func=main)


def main(args):
    if args.open_ended_questions and args.model_name in ENCODER_MODELS:
        raise ValueError(
            f"Open ended questions are not supported by encoder models: {ENCODER_MODELS}."
        )
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
        civet_collator = Collator(processor)  # type: ignore
    elif args.model_name == "CoCa":
        model, preprocess = Model(pretrained=model_name, device=args.device)
        tokenizer = open_clip.get_tokenizer("coca_ViT-L-14")
        civet_collator = Collator(preprocess, tokenizer)
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

            civet_collator = Collator(processor, args.instruction)
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

    predicted_samples = []
    if (
        os.path.exists(os.path.join(args.out_dir, f"predicted_samples.json"))
        and not args.override
    ):
        with open(os.path.join(args.out_dir, f"predicted_samples.json"), "r") as f:
            predicted_samples = json.load(f)

    test_ds = CIVETDataset(
        data_folder=args.data_folder,
        predicted_samples=set(predicted_samples),
        debug=args.debug,
        open_ended_questions=args.open_ended_questions,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1 if is_encoder else args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=civet_collator,
    )

    results = {}
    if os.path.exists(os.path.join(args.out_dir, f"results.json")):
        with open(os.path.join(args.out_dir, f"results.json"), "r") as f:
            results = json.load(f)

    current_step = len(predicted_samples)
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

                if is_encoder:
                    input = input_texts[pred.item()]  # type: ignore
                    original_pred = input
                    pred = match_number_in_text(input)
                else:
                    input = processor.decode(input_texts.input_ids[i], skip_special_tokens=True)  # type: ignore
                    original_pred = pred
                    pred = match_number_in_text(pred)  # type: ignore

                res = {
                    "target": target,
                    "input": input,
                    "original_pred": original_pred,
                    "pred": pred,
                }

                img_id, ent_type, *q_type = sample_id.split("-")

                if img_id not in results:
                    results[img_id] = {}

                if ent_type not in results[img_id]:
                    results[img_id][ent_type] = {}

                if len(q_type) == 2:
                    q_type, sub_q_type = q_type
                    if q_type not in results[img_id][ent_type]:
                        results[img_id][ent_type][q_type] = {}

                    results[img_id][ent_type][q_type][sub_q_type] = res
                elif len(q_type) == 1:
                    # unpack it
                    q_type = q_type[0]
                    results[img_id][ent_type][q_type] = res
                else:
                    raise ValueError(f"Invalid question type: {q_type}")

                current_step += 1
                if current_step % args.save_every == 0:
                    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
                        json.dump(results, f, indent=4)

                    with open(
                        os.path.join(args.out_dir, f"predicted_samples.json"), "w"
                    ) as f:
                        json.dump(predicted_samples, f, indent=4)

                # add the predicted samples to keep track of them
                predicted_samples.append(sample_id)

    with open(os.path.join(args.out_dir, f"results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(args.out_dir, f"predicted_samples.json"), "w") as f:
        json.dump(predicted_samples, f, indent=4)

    y_true = []
    y_pred = []
    summary = {}
    results_per_ent_type = aggregate_results_per_ent_type(results)
    for ent_type in results_per_ent_type:
        ent_y_true = results_per_ent_type[ent_type]["y_true"]
        ent_y_pred = results_per_ent_type[ent_type]["y_pred"]

        report = create_classification_report(
            ent_y_true, ent_y_pred, f"{args.out_dir}/{ent_type}"
        )

        create_confusion_matrix(ent_y_true, ent_y_pred, f"{args.out_dir}/{ent_type}")

        y_true.extend(ent_y_true)
        y_pred.extend(ent_y_pred)
        summary[ent_type] = {"accuracy": report["accuracy"]}

    create_classification_report(y_true, y_pred, args.out_dir)

    create_confusion_matrix(y_true, y_pred, args.out_dir)

    with open(os.path.join(args.out_dir, f"summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
