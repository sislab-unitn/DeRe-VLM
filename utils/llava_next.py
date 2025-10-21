from transformers import LlavaNextProcessor


class GenerationCollator:
    def __init__(
        self,
        processor: LlavaNextProcessor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token  # type: ignore

        samples = []
        questions = []
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

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                        {"type": "image"},
                    ],
                },
            ]

            sample = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,  # used in the huggingface example
            )
            samples.append(sample)

        input_ids = self.processor(
            text=samples, images=images, return_tensors="pt", padding=True
        )
        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids
