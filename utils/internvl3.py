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

        input_ids = self.processor.apply_chat_template(
            conversations,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids
