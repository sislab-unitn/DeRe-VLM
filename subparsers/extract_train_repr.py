import argparse
import os
import pickle
import random
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor


from utils import qwen2_vl, llava_next, paligemma
from utils.data import CIVETDataset, PixmoCountDataset
from utils.utils import ENCODER_MODELS, MAP_MODELS, get_encoder_repr


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "extract-train-repr",
        help="Extract the hidden representation of the last token to train the output layer.",
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
        "--instruction",
        metavar="INSTR",
        type=str,
        default="Answer using as few words as possible",
        help="Instruction used for generation (encoder-decoder only).",
    )
    parser.add_argument(
        "--dataset-type",
        metavar="DATASET_TYPE",
        type=str,
        default="civet",
        choices=["civet", "pixmo-count"],
        help="Type of dataset to use.",
    )

    parser.set_defaults(func=main)


def main(args):
    # get the model and processor based on the model name
    model_name, Model, Collator = MAP_MODELS[args.model_name]

    if args.model_name in ENCODER_MODELS:
        raise ValueError(
            f"Model {args.model_name} is an encoder model. This parsers only works with LLM-based VLM."
        )
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

    if args.dataset_type == "civet":
        test_ds = CIVETDataset(
            data_folder=args.data_folder,
            predicted_samples=set(),
            debug=args.debug,
        )
    elif args.dataset_type == "pixmo-count":
        test_ds = PixmoCountDataset(data_folder=args.data_folder)
    else:
        raise ValueError(f"Invalid dataset type: {args.dataset_type}")

    test_loader = DataLoader(
        test_ds,
        batch_size=1 if is_encoder else args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=civet_collator,
    )

    with open(os.path.join(args.out_dir, "representations.pkl"), "ab") as f:
        with torch.no_grad():
            for input_ids, _, ent_y_true, sample_ids in tqdm(
                test_loader, desc=f"Counting Objects"
            ):
                _, batch_hidden_states = get_encoder_repr(
                    input_ids, model, args.model_name, args.device
                )

                batch_token_ids = input_ids.input_ids
                for sample_id, token_ids, hidden_states, target in zip(
                    sample_ids,
                    batch_token_ids,
                    batch_hidden_states,
                    ent_y_true,
                ):

                    last_hidden_state = hidden_states[-1]
                    last_token = last_hidden_state[-1, :]

                    res = {
                        "sample_id": sample_id,
                        "token_ids": token_ids.cpu(),
                        "last_token_repr": last_token.detach().clone(),
                        "target": target,
                    }

                    pickle.dump(res, f)
