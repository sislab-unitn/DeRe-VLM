import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import datasets
from PIL import Image
from torch.utils.data import Dataset


class CIVETDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        predicted_samples: Set[str],
        debug: bool = False,
        open_ended_questions: bool = False,
        ids_to_process: Optional[Set[str]] = None,
    ):
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        with open(os.path.join(data_folder, "questions.json"), "r") as f:
            data: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        with open(os.path.join(data_folder, "answers.json"), "r") as f:
            answers: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        if debug:
            random_ids = random.sample(list(data.keys()), 100)

        for img_id, entities in data.items():
            # Skip samples not in the subset of ids when debugging
            if debug and img_id not in random_ids:
                continue

            # Skip samples not in the subset of ids to process
            if ids_to_process and img_id not in ids_to_process:
                continue

            for ent_type, question_types in entities.items():
                for q_type, question in question_types.items():
                    if isinstance(question, dict):
                        for sub_q_type, sub_question in question.items():
                            if open_ended_questions:
                                sub_question = self._make_open_ended(sub_question)
                            answer = answers[img_id][ent_type][q_type][sub_q_type]
                            key = f"{img_id}-{ent_type}-{q_type}-{sub_q_type}"
                            if key not in predicted_samples:
                                self._add_sample(
                                    sample_id=key,
                                    question=sub_question,
                                    answer=answer,
                                    data_folder=data_folder,
                                    img_id=img_id,
                                )
                    else:
                        if open_ended_questions:
                            question = self._make_open_ended(question)
                        answer = answers[img_id][ent_type][q_type]
                        key = f"{img_id}-{ent_type}-{q_type}"
                        if key not in predicted_samples:
                            self._add_sample(
                                sample_id=key,
                                question=question,
                                answer=answer,
                                data_folder=data_folder,
                                img_id=img_id,
                            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        return (
            Image.open(self.images[idx]),
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )

    def _add_sample(
        self, sample_id: str, question: str, answer: str, data_folder: str, img_id: str
    ):
        self.sample_ids.append(sample_id)
        self.questions.append(question)
        self.answers.append(answer)
        self.images.append(os.path.join(data_folder, "images", f"{img_id}.png"))

    def _make_open_ended(self, question: str) -> str:
        question = re.sub(r"Choose from \[.*\].", "", question).strip()
        assert (
            "[" not in question and "]" not in question
        ), f"squared brackets not removed from {question}"

        return question


class CountBenchDataset(Dataset):
    def __init__(
        self,
        is_encoder: bool = False,
    ):
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        count_bench_qa = datasets.load_dataset("vikhyatk/CountBenchQA", split="test")

        for q_id, sample in enumerate(count_bench_qa):
            sample_id = f"CountBenchQA-{q_id}"

            if is_encoder:
                self.questions.append(sample["text"])
            else:
                self.questions.append(sample["question"])

            self.sample_ids.append(sample_id)
            self.answers.append(sample["number"])
            self.images.append(sample["image"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        return (
            self.images[idx],
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )


class PixmoCountDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        range: List[int] = list(range(0, 10)),
    ):
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        with open(f"{data_folder}/samples.json", "r") as f:
            samples = json.load(f)

        for q_id, sample in enumerate(samples):
            sample_id = f"pixmo_count-{q_id}"

            answer = sample["count"]

            if answer in range:
                category = sample["label"]
                self.questions.append(f"How many {category} are there?")
                self.sample_ids.append(sample_id)
                self.answers.append(answer)
                image_path = os.path.join(data_folder, "images", sample["image_path"])
                self.images.append(image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        return (
            Image.open(self.images[idx]),
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )
