import pickle
import random
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import open_clip
import torch
from transformers import (
    CLIPModel,
    LlavaForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModelForImageTextToText,
)
from transformers.modeling_utils import PreTrainedModel

from utils.clip import SURECollator as CLIPCollator
from utils.coca import SURECollator as CoCaCollator
from utils.llava_next import GenerationCollator as LLavaNextGenerationCollator
from utils.paligemma import GenerationCollator as PaliGemmaGenerationCollator
from utils.qwen2_vl import GenerationCollator as Qwen2VLGenerationCollator
from utils.internvl3 import GenerationCollator as InternVL3GenerationCollator


MAP_MODELS = {
    "CLIP": (
        "openai/clip-vit-large-patch14-336",
        CLIPModel.from_pretrained,
        CLIPCollator,
    ),
    "CoCa": (
        "mscoco_finetuned_laion2B-s13B-b90k",
        partial(
            open_clip.create_model_from_pretrained,
            model_name="coca_ViT-L-14",
        ),
        CoCaCollator,
    ),
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        Qwen2VLGenerationCollator,
    ),
    "llava-next-interleave-7b": (
        "llava-hf/llava-interleave-qwen-7b-dpo-hf",
        partial(
            LlavaForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        LLavaNextGenerationCollator,
    ),
    "paligemma2-10b": (
        "google/paligemma2-10b-mix-448",
        partial(
            PaliGemmaForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        PaliGemmaGenerationCollator,
    ),
    "llava-onevision-7b": (
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        partial(
            LlavaOnevisionForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        LLavaNextGenerationCollator,
    ),
    "internvl3-8b": (
        "OpenGVLab/InternVL3-8B-hf",
        partial(
            AutoModelForImageTextToText.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        InternVL3GenerationCollator,
    ),
}

ENCODER_MODELS = ["CLIP", "CoCa"]


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)  # This will return each dict
            except EOFError:
                break


def get_encoder_repr(
    input_ids, model, model_name: str, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_name == "CLIP":
        input_ids.to(model.device)
        outputs = model(**input_ids)
        image_features = outputs.image_embeds
        hidden_states = outputs.vision_model_output.last_hidden_state
        hidden_states = hidden_states.detach().to("cpu")
    elif model_name == "CoCa":
        image, text = input_ids
        image = image.to(device)
        text = text.to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        cls, hidden_states = model.visual(image)
        cls = cls.unsqueeze(1)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = torch.cat((cls, hidden_states), dim=0)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states.detach().to("cpu")
    else:
        if model_name == "internvl3-8b":
            input_ids.to(dtype=model.dtype)
        input_ids.to(model.device)
        outputs = model(**input_ids, output_hidden_states=True)

        if "llava" in model_name or model_name in ["paligemma2-10b", "internvl3-8b"]:
            image_features = outputs.image_hidden_states
        elif model_name == "qwen2-vl-7b":
            pixel_values = input_ids.pixel_values
            image_grid_thw = input_ids.image_grid_thw

            pixel_values = pixel_values.type(model.visual.get_dtype())
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)

            image_features = image_embeds.to(outputs.logits.dtype)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        image_features = image_features.reshape(
            input_ids.input_ids.shape[0], -1, image_features.shape[-1]
        )
        # ignore the embedding layer
        hidden_states = torch.stack(
            [h_s.detach().to("cpu") for h_s in outputs.hidden_states[1:]], dim=1
        )

    image_features = image_features.detach().to("cpu")

    return image_features, hidden_states


def classify(input_ids, model, model_name: str, device) -> torch.Tensor:
    if model_name == "CLIP":
        input_ids.to(model.device)
        outputs = model(**input_ids)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        prob = logits_per_image.softmax(
            dim=-1
        )  # we can take the softmax to get the label probabilities
    elif model_name == "CoCa":
        image, text = input_ids
        image = image.to(device)
        text = text.to(device)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        prob = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return prob


def generate(
    input_ids,
    model: PreTrainedModel,
    model_name: str,
    processor,
    max_new_tokens: int,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[str]:
    if model_name in ["paligemma2-10b", "internvl3-8b"] and "input_ids" in input_ids:
        input_ids.to(dtype=model.dtype)

    input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            do_sample=False,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,  # type: ignore
        )

    if "llava" in model_name or model_name in ["paligemma2-10b", "internvl3-8b"]:
        if "input_ids" in input_ids:
            output = output[:, input_ids.input_ids.size(-1) :]
    elif model_name == "qwen2-vl-7b":
        if "input_ids" in input_ids:
            output = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(input_ids.input_ids, output)
            ]

    generated_answers = processor.batch_decode(output, skip_special_tokens=True)

    return generated_answers


def predict(
    input_ids,
    model: PreTrainedModel,
    model_name: str,
    processor,
    valid_tokens: List[str],
) -> Tuple[List[str], List[bool]]:
    if model_name == "paligemma2-10b":
        input_ids.to(torch.bfloat16)

    input_ids.to(model.device)
    output = model(**input_ids)
    logits = output.logits

    tokenizer = processor.tokenizer
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in valid_tokens]
    assert (
        None not in token_ids
    ), f"Among the tokens {valid_tokens}, it was not possible to convert some of them {token_ids}"
    token_ids = torch.tensor(token_ids, dtype=torch.long)

    last_token_logits = logits[:, -1, :]
    next_tokens = last_token_logits.cpu().argmax(dim=-1)
    next_valid_tokens_idx = last_token_logits[:, token_ids].cpu().argmax(dim=-1)
    next_valid_tokens = token_ids[next_valid_tokens_idx]
    different_preds = (next_tokens != next_valid_tokens).tolist()

    preds = tokenizer.batch_decode(next_valid_tokens)

    return preds, different_preds


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
