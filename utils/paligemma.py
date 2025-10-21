from typing import List, Tuple

import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.utils.import_utils import is_torchdynamo_compiling


class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        questions = []
        texts = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            # Solve the issue of L images
            if image.mode != "RGB":
                image = image.convert("RGB")

            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

            prompt = "<image> answer en"
            if self.instruction:
                prompt = f"{prompt} {self.instruction}"
            prompt = f"{prompt} {question}"

            texts.append(prompt)

        input_ids = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


def prepare_inputs_embeds(
    input_ids, model: PaliGemmaForConditionalGeneration
) -> torch.Tensor:

    input_ids = input_ids.to(model.device)

    pixel_values = input_ids.pixel_values
    input_ids = input_ids.input_ids

    # code from PaliGemmaForConditionalGeneration's forward method
    inputs_embeds = model.get_input_embeddings()(input_ids)

    image_features = model.get_image_features(pixel_values)

    special_image_mask = (input_ids == model.config.image_token_index).unsqueeze(-1)
    special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
        inputs_embeds.device
    )
    if (
        not is_torchdynamo_compiling()
        and inputs_embeds[special_image_mask].numel() != image_features.numel()
    ):
        image_tokens_in_text = torch.sum(input_ids == model.config.image_token_index)
        raise ValueError(
            f"Number of images does not match number of special image tokens in the input text. "
            f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
            "tokens from image embeddings."
        )
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    return inputs_embeds


class TrainingCollator:
    def __init__(
        self,
        processor: PaliGemmaProcessor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):
        self.processor.tokenizer.padding_side = "right"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        questions = []
        texts = []
        images = []
        answers = []
        suffixes = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            # Solve the issue of L images
            if image.mode != "RGB":
                image = image.convert("RGB")

            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

            prompt = "<image> answer en"
            if self.instruction:
                prompt = f"{prompt} {self.instruction}"
            prompt = f"{prompt} {question}"

            texts.append(prompt)
            suffixes.append(f"{answer}")

        input_ids = self.processor(
            text=texts,
            images=images,
            suffix=suffixes,  # type: ignore
            padding=True,
            return_tensors="pt",
        )

        return input_ids, questions, answers, sample_ids
