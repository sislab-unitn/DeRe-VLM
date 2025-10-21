import argparse
import os
import pickle
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
from utils.utils import ENCODER_MODELS, MAP_MODELS, get_encoder_repr


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "extract-layer-repr",
        help="Extract different representations from the model at each layer.",
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

    parser.set_defaults(func=main)


def extract_layer_info(
    model, processor, res, hs_i, visual_token_mask, layer_num, args
) -> dict:
    # average across the sequence length
    res[f"dec_hidden_states_layer_{layer_num+1}"] = hs_i.mean(dim=0).float().numpy()

    # get the last token
    last_token = hs_i[-1, :]
    res[f"dec_last_token_layer_{layer_num+1}"] = last_token.float().numpy()

    # get prediction for the last token
    last_token = last_token.to(model.device)

    if args.model_name in ["qwen2-vl-7b", "internvl3-8b"]:
        out = model.lm_head(last_token)
    elif "llava" in args.model_name or args.model_name == "paligemma2-10b":
        out = model.language_model.lm_head(last_token)
    else:
        raise NotImplementedError(
            f"Model {args.model_name} not implemented for prediction."
        )

    pred = processor.tokenizer.decode(  # type: ignore
        out.detach().cpu().argmax().tolist()
    )
    res[f"pred_layer_{layer_num+1}"] = pred

    visual_tokens = hs_i[visual_token_mask]
    # get the mean of the decoder image features
    res[f"dec_image_features_layer_{layer_num+1}"] = (
        visual_tokens.mean(dim=0).float().numpy()
    )
    # get the last visual token
    res[f"dec_image_last_token_layer_{layer_num+1}"] = (
        visual_tokens[-1, :].float().numpy()
    )

    return res


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

        if args.model_name == "qwen2-vl-7b":
            processor = Processor(
                min_pixels=100 * 28 * 28,
                max_pixels=2304 * 28 * 28,  # size of 1344x1344 images
            )
            image_token_id = 151655
        elif "llava" in args.model_name:
            processor = Processor()
            image_token_id = 151646
        elif args.model_name == "paligemma2-10b":
            processor = Processor()
            torch.set_float32_matmul_precision("high")
            image_token_id = 257152
        elif args.model_name == "internvl3-8b":
            processor = Processor()
            image_token_id = 151667
        else:
            raise ValueError(f"Invalid model name: {args.model_name}")

        civet_collator = Collator(processor, args.instruction)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

    # model config
    model.eval()

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_encoder = args.model_name in ENCODER_MODELS

    test_ds = CIVETDataset(
        data_folder=args.data_folder,
        predicted_samples=set(),
        debug=args.debug,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1 if is_encoder else args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=civet_collator,
    )

    with open(os.path.join(args.out_dir, "predictions.pkl"), "ab") as f:
        with torch.no_grad():
            for input_ids, _, ent_y_true, sample_ids in tqdm(
                test_loader, desc=f"Counting Objects"
            ):
                ent_y_pred, batch_hidden_states = get_encoder_repr(
                    input_ids, model, args.model_name, args.device
                )

                if len(ent_y_pred.shape) != 2:
                    if len(ent_y_pred.shape) == 3:
                        ent_y_pred = ent_y_pred.mean(dim=1).float().numpy()
                    else:
                        raise ValueError(
                            f"Unexpected image features shape: {ent_y_pred.shape}"
                        )

                batch_token_ids = input_ids.input_ids
                for sample_id, token_ids, img_features, hidden_states, target in zip(
                    sample_ids,
                    batch_token_ids,
                    ent_y_pred,
                    batch_hidden_states,
                    ent_y_true,
                ):
                    # compute the mask on the token ids to extract the visual tokens from the hidden states
                    visual_token_mask = token_ids == image_token_id
                    visual_token_mask = visual_token_mask.to(batch_hidden_states.device)

                    res = {
                        "sample_id": sample_id,
                        "enc_image_features": img_features,
                        "target": target,
                    }

                    if args.model_name not in ENCODER_MODELS:
                        for i, hs_i in enumerate(hidden_states):
                            res = extract_layer_info(
                                model=model,
                                processor=processor,
                                res=res,
                                hs_i=hs_i,
                                visual_token_mask=visual_token_mask,
                                layer_num=i,
                                args=args,
                            )

                    pickle.dump(res, f)
