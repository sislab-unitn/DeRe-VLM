import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration


class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        questions = []
        conversations = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)  # type: ignore
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


def prepare_inputs_embeds(
    input_ids, model: Qwen2VLForConditionalGeneration
) -> torch.Tensor:
    input_ids = input_ids.to(model.device)

    pixel_values = input_ids.pixel_values
    image_grid_thw = input_ids.image_grid_thw
    input_ids = input_ids.input_ids

    # code from Qwen2VLForConditionalGeneration's forward method
    inputs_embeds = model.model.embed_tokens(input_ids)
    pixel_values = pixel_values.type(model.visual.get_dtype())
    image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
    n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )
    image_mask = (
        (input_ids == model.config.image_token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds
