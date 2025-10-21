import re
from typing import List, Tuple

from PIL import Image


class SURECollator:
    def __init__(self, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):

        assert len(batch) == 1, "SURECollator only supports batch size of 1"
        image, question, answer, sample_id = batch[0]
        _, ent_type, *_ = sample_id.split("-")
        ent = " ".join(ent_type.split("_"))

        # exract the candidates
        candidates = re.search(r"\[.*\]", question)
        assert candidates is not None, f"candidates not found in '{question}'"
        candidates = candidates.group()[1:-1]
        assert (
            "[" not in candidates and "]" not in candidates
        ), f"squared brackets not removed from {candidates}"

        candidates = [int(c) for c in candidates.split(",")]
        candidates.sort()

        # create a candidate by combining the question and the candidate
        texts = [f"a photo of {n} {ent}" for n in candidates]

        text = self.tokenizer(texts)
        image = self.preprocess(image).unsqueeze(0)

        return (image, text), texts, [answer], [sample_id]


class CountBenchCollator:
    def __init__(self, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def _create_candidates(self, question: str) -> List[str]:
        number = re.search(
            r"two|three|four|five|six|seven|eight|nine|ten",
            question,
            flags=re.IGNORECASE,
        )

        assert number is not None, f"candidates not found in '{question}'"
        number = number.group()

        candidates = []
        for label in [
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]:
            candidates.append(question.replace(number, label))

        return candidates

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):

        assert len(batch) == 1, "CountBenchCollator only supports batch size of 1"
        image, question, answer, sample_id = batch[0]

        candidates = self._create_candidates(question)
        text = self.tokenizer(candidates)
        image = self.preprocess(image).unsqueeze(0)

        return (image, text), candidates, [answer], [sample_id]
