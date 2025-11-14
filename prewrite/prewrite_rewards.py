# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Callable

from typing import Optional

from trl.import_utils import is_math_verify_available

if is_math_verify_available():
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

import requests
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import random
import numpy as np
import time
from sklearn.metrics import f1_score as sklearn_f1_score

import asyncio

import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge
from sglang.srt.entrypoints.engine import Engine

gsm8k_dataset_path = "/root/group-shared/jrc/data/gsm8k"
gsm8k_dataset_config = "main"
ag_news_dataset_path = "/root/group-shared/jrc/data/ag_news"
ag_news_dataset_config = "default"
sst2_dataset_path = "/root/group-shared/jrc/data/sst2"
sst2_dataset_config = "default"
nq_open_dataset_path = "/root/group-shared/jrc/data/nq_open"
nq_open_dataset_config = "nq_open"
math_500_dataset_path = "/root/group-shared/jrc/data/MATH-500"
math_500_dataset_config = "default"
copa_dataset_path = "/root/group-shared/jrc/data/super_glue"
copa_dataset_config = "copa"
boolq_dataset_path = "/root/group-shared/jrc/data/super_glue"
boolq_dataset_config = "boolq"
cb_dataset_path = "/root/group-shared/jrc/data/super_glue"
cb_dataset_config = "cb"

task_classification_map = {
    "ag_news": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Tech",
    },
    "sst2": {
        0: "negative",
        1: "positive",
    },
    "copa": {
        0: "1",
        1: "2",
    },
    "boolq": {
        0: "False",
        1: "True",
    },
    "axb": {
        0: "ent",
        1: "not",
    },
    "cb": {
        0: "ent",
        1: "contr",
        2: "neutral",
    },
}

task_token_ids = {
    "ag_news": [10134, 40979, 22727, 34097],
    "sst2": [30487, 42224],
    "copa": [16, 17],
    "boolq": [4049, 2514],
    "cb": [306, 8222, 59568],
}

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

import unicodedata
import re
import string

def normalize_answer(s):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(answer: str, completion: str) -> float:
    """
    Calculate word-level F1 score between two strings.
    
    Args:
        answer: Ground truth answer string
        completion: Predicted completion string
    
    Returns:
        F1 score (float between 0 and 1)
    
    Example:
        >>> f1_score("the quick brown fox", "the brown fox")
        0.8  # precision=3/3=1.0, recall=3/4=0.75, f1=2*1.0*0.75/(1.0+0.75)=0.857
    """
    # Normalize both strings
    answer_normalized = normalize_answer(answer)
    completion_normalized = normalize_answer(completion)
    
    # Split into words
    answer_tokens = answer_normalized.split()
    completion_tokens = completion_normalized.split()
    
    # Edge cases
    if len(answer_tokens) == 0 and len(completion_tokens) == 0:
        return 1.0
    if len(answer_tokens) == 0 or len(completion_tokens) == 0:
        return 0.0
    
    # Convert to multisets (Counter) to handle repeated words
    from collections import Counter
    answer_bag = Counter(answer_tokens)
    completion_bag = Counter(completion_tokens)
    
    # Calculate intersection (common words with their minimum counts)
    common = answer_bag & completion_bag
    num_common = sum(common.values())
    
    # Calculate precision and recall
    precision = num_common / len(completion_tokens)
    recall = num_common / len(answer_tokens)
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1
  
train_dataset_dict = {
    "gsm8k": load_dataset(gsm8k_dataset_path, gsm8k_dataset_config)["train"],
    "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config)["train"],
    "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config)["train"],
    "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config)["train"],
    "MATH-500": load_dataset(math_500_dataset_path, math_500_dataset_config)["test"],
    "copa": load_dataset(copa_dataset_path, copa_dataset_config)["train"],
    "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config)["train"],
    "cb": load_dataset(cb_dataset_path, cb_dataset_config)["train"],
}

test_dataset_dict = {
    "gsm8k": load_dataset(gsm8k_dataset_path, gsm8k_dataset_config)["test"],
    "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config)["test"],
    "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config)["validation"],
    "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config)["validation"],
    "MATH-500": load_dataset(math_500_dataset_path, math_500_dataset_config)["test"],
    "copa": load_dataset(copa_dataset_path, copa_dataset_config)["validation"],
    "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config)["validation"],
    "cb": load_dataset(cb_dataset_path, cb_dataset_config)["validation"],
}

def get_messages(instruction: str, example: dict, dataset_name: str, system_prompt: str="") -> list[dict[str, str]]:
    if dataset_name == "gsm8k":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nQuestion: {example['question']}"}]
    elif dataset_name == "MATH-500":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nProblem: {example['problem']}"}]
    elif dataset_name == "ag_news":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nText: {example['text']}"}]
    elif dataset_name == "sst2":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nSentence: {example['sentence']}"}]
    elif dataset_name == "copa":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nPremise: {example['premise']}\nChoice1: {example['choice1']}\nChoice2: {example['choice2']}\nQuestion: {example['question']}"}]
    elif dataset_name == "boolq":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nPassage: {example['passage']}\nQuestion: {example['question']}"}]
    elif dataset_name == "cb":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nPremise: {example['premise']}\nHypothesis: {example['hypothesis']}"}]
    elif dataset_name == "nq_open":
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{instruction}\nQuestion: {example['question']}"}]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

def get_prewrite(generate_model_ip: str, generate_model_port: int, generate_model_config: dict, tokenizer: AutoTokenizer, system_prompt: str="") -> Callable:

    def prewrite_accuracy_reward(completions: list[list[dict[str, str]]], answer: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that checks if the completion is the same as the ground truth.
            - If both gold and prediction are parseable → use math verification.
            - If not parseable → compare as normalized text.

        Args:
            completions (`list[list[dict[str, str]]]`):
                List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
                containing the key `"content"` with the value being the text of the completion.
            solution: (`list[str]`):
                List of the raw-text solutions to the questions/problems/prompts.
            **kwargs:
                Additional keyword arguments. This function does not use them, but they are required in the function
                signature to ensure compatibility with trainers like [`GRPOTrainer`].
        Example:
        ```python
        >>> from trl.rewards import accuracy_reward

        >>> solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        >>> completion = [
        ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{3}}"}],
        ...     [{"role": "assistant", "content": r"My answer is \boxed{\frac{1}{2}}"}],
        ... ]
        >>> accuracy_reward(completion, solution)
        [1.0, 0.0]
        ```
        """
        if not is_math_verify_available():
            raise ImportError("Please install the `math_verify` package to use accuracy_reward")

        contents = []
        for completion in completions:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": completion}], 
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            response = requests.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": generate_model_config,
                },
            )
            contents.append(response.json()["text"])
        rewards = []
        for content, sol in zip(contents, answer):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
            )
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Compute binary rewards if verifiable, `None` otherwise to skip this example
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = None
            else:
                # If the gold solution is not parseable, we assign `None` to skip this example
                reward = float(content.strip().lower() == sol.strip().lower())
            rewards.append(reward)

        return rewards
    
    return prewrite_accuracy_reward

def get_accuracy_math(generate_model_ip: str, generate_model_port: int, tokenizer: AutoTokenizer, system_prompt: str="", test_size: int=10, is_test: bool=False) -> Callable:

    def accuracy_math_reward(completions: list[list[dict[str, str]]], dataset_name: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that checks if the completion is the same as the ground truth.
        """
        rewards = []
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name in ["gsm8k", "MATH-500"]:
                if is_test:
                    dataset = test_dataset_dict[single_dataset_name]
                    sample_size = len(dataset)
                else:
                    dataset = train_dataset_dict[single_dataset_name]
                    sample_size = min(test_size, len(dataset))
                scores = []
                generate_model_config = {
                    "temperature": 0.0,
                    "max_new_tokens": 2048,
                }
                random.seed(time.time())
                indices = random.sample(range(len(dataset)), sample_size)
                prompts = []
                for example in dataset.select(indices):
                    prompt = tokenizer.apply_chat_template(
                        get_messages(instruction, example, single_dataset_name, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(prompt)
                response = requests.post(
                    f"http://{generate_model_ip}:{generate_model_port}/generate",
                    json={
                        "text": prompts,
                        "sampling_params": generate_model_config,
                    },
                )
                completions = [output["text"] for output in response.json()]
                for example, completion in zip(dataset.select(indices), completions):
                    gold_parsed = parse(
                        example["answer"],
                        extraction_mode="first_match",
                    )
                    if len(gold_parsed) != 0:
                        # We require the answer to be provided in correct latex (no malformed operators)
                        answer_parsed = parse(
                            completion,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False,
                                        malformed_operators=False,
                                        basic_latex=True,
                                        boxed="all",
                                        units=True,
                                    ),
                                    # Ensures that boxed is tried first
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        # Compute binary rewards if verifiable, `None` otherwise to skip this example
                        try:
                            score = float(verify(gold_parsed, answer_parsed))
                        except Exception:
                            score = None
                    else:
                        # If the gold solution is not parseable, we assign `None` to skip this example
                        score = float(completion.strip().lower() == example["answer"].strip().lower())
                    scores.append(score)
                rewards.append(np.mean(scores))
            else:
                rewards.append(0)
        return rewards
    return accuracy_math_reward

def get_accuracy_classification(generate_model_ip: str, generate_model_port: int, tokenizer: AutoTokenizer, system_prompt: str="", test_size: int=10,  custom_logit_processor: CustomLogitProcessor=None, is_test: bool=False) -> Callable:

    def accuracy_classification_reward(completions: list[list[dict[str, str]]], dataset_name: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that checks if the completion is the same as the ground truth.
        """
        rewards = []
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name in ["ag_news", "sst2", "copa", "boolq", "cb"]:
                if is_test:
                    dataset = test_dataset_dict[single_dataset_name]
                    sample_size = len(dataset)
                else:
                    dataset = train_dataset_dict[single_dataset_name]
                    sample_size = min(test_size, len(dataset))
                scores = []
                generate_model_config = {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "custom_params": {"token_ids": task_token_ids[single_dataset_name]},
                }
                classification_map = task_classification_map[single_dataset_name]
                random.seed(time.time())
                indices = random.sample(range(len(dataset)), sample_size)
                prompts = []
                for example in dataset.select(indices):
                    prompt = tokenizer.apply_chat_template(
                        get_messages(instruction, example, single_dataset_name, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(prompt)
                response = requests.post(
                    f"http://{generate_model_ip}:{generate_model_port}/generate",
                    json={
                        "text": prompts,
                        "sampling_params": generate_model_config,
                        "custom_logit_processor": custom_logit_processor.to_str(),
                    },
                )
                completions = [output["text"] for output in response.json()]
                for example, completion in zip(dataset.select(indices), completions):
                    if completion == classification_map[example["label"]]:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                rewards.append(np.mean(scores))
            else:
                rewards.append(0)
        return rewards
    return accuracy_classification_reward

def get_accuracy_nq_open_instruction_prewrite(generate_model_ip: str, generate_model_port: int, tokenizer: AutoTokenizer, system_prompt: str="", test_size: int=10, is_test: bool=False) -> Callable:

    def accuracy_nq_open_instruction_prewrite(completions: list[list[dict[str, str]]], dataset_name: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that checks if the completion is the same as the ground truth.
        """
        rewards = []
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name == "nq_open":
                if is_test:
                    dataset = test_dataset_dict[single_dataset_name]
                    sample_size = len(dataset)
                else:
                    dataset = train_dataset_dict[single_dataset_name]
                    sample_size = min(test_size, len(dataset))
                generate_model_config = {
                    "temperature": 0.0,
                    "max_new_tokens": 100,
                }
                scores = []
                random.seed(time.time())
                indices = random.sample(range(len(dataset)), sample_size)
                prompts = []
                for example in dataset.select(indices):
                    prompt = tokenizer.apply_chat_template(
                        get_messages(instruction, example, single_dataset_name, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(prompt)
                response = requests.post(
                    f"http://{generate_model_ip}:{generate_model_port}/generate",
                    json={
                        "text": prompts,
                        "sampling_params": generate_model_config,
                    },
                )
                completions = [output["text"] for output in response.json()]
                for example, completion in zip(dataset.select(indices), completions):
                    flag = False
                    for answer in example["answer"]:
                        if normalize_answer(completion) == normalize_answer(answer):
                            flag = True
                            break
                    if flag:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                rewards.append(np.mean(scores))
            else:
                rewards.append(0)
        return rewards
    return accuracy_nq_open_instruction_prewrite

def get_f1_nq_open_instruction_prewrite(generate_model_ip: str, generate_model_port: int, tokenizer: AutoTokenizer, system_prompt: str="", test_size: int=10, is_test: bool=False) -> Callable:

    def f1_nq_open_instruction_prewrite(completions: list[list[dict[str, str]]], dataset_name: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that checks if the completion is the same as the ground truth.
        """
        rewards = []
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name == "nq_open":
                if is_test:
                    dataset = test_dataset_dict[single_dataset_name]
                    sample_size = len(dataset)
                else:
                    dataset = train_dataset_dict[single_dataset_name]
                    sample_size = min(test_size, len(dataset))
                generate_model_config = {
                    "temperature": 0.0,
                    "max_new_tokens": 100,
                }
                scores = []
                random.seed(time.time())
                indices = random.sample(range(len(dataset)), sample_size)
                prompts = []
                for example in dataset.select(indices):
                    prompt = tokenizer.apply_chat_template(
                        get_messages(instruction, example, single_dataset_name, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(prompt)
                response = requests.post(
                    f"http://{generate_model_ip}:{generate_model_port}/generate",
                    json={
                        "text": prompts,
                        "sampling_params": generate_model_config,
                    },
                )
                completions = [output["text"] for output in response.json()]
                for example, completion in zip(dataset.select(indices), completions):
                    score = 0
                    for answer in example["answer"]:
                        score = max(score, f1_score(answer, completion))
                    scores.append(score)
                rewards.append(np.mean(scores))
            else:
                rewards.append(0)
        return rewards   
    return f1_nq_open_instruction_prewrite

def get_f1_classification(generate_model_ip: str, generate_model_port: int, tokenizer: AutoTokenizer, system_prompt: str="", test_size: int=10,  custom_logit_processor: CustomLogitProcessor=None, is_test: bool=False) -> Callable:

    def f1_classification_reward(completions: list[list[dict[str, str]]], dataset_name: list[str], **kwargs) -> list[Optional[float]]:
        r"""
        Reward function that calculates F1 score for AG News classification.
        Uses macro-F1 (average F1 across all classes).
        """
        rewards = []
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name in ["ag_news", "sst2", "copa", "boolq", "cb"]:
                if is_test:
                    dataset = test_dataset_dict[single_dataset_name]
                    sample_size = len(dataset)
                else:
                    dataset = train_dataset_dict[single_dataset_name]
                    sample_size = min(test_size, len(dataset))
                generate_model_config = {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "custom_params": {"token_ids": task_token_ids[single_dataset_name]},
                }
                classification_map = task_classification_map[single_dataset_name]
                # Reverse mapping for converting predictions back to label IDs
                reverse_map = {v: k for k, v in classification_map.items()}
                
                # Collect all predictions and ground truths
                predictions = []
                ground_truths = []                
                random.seed(time.time())
                indices = random.sample(range(len(dataset)), sample_size)
                prompts = []
                for example in dataset.select(indices):
                    prompt = tokenizer.apply_chat_template(
                        get_messages(instruction, example, single_dataset_name, system_prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    prompts.append(prompt)
                response = requests.post(
                    f"http://{generate_model_ip}:{generate_model_port}/generate",
                    json={
                        "text": prompts,
                        "sampling_params": generate_model_config,
                        "custom_logit_processor": custom_logit_processor.to_str(),
                    },
                )
                completions = [output["text"] for output in response.json()]
                for example, completion in zip(dataset.select(indices), completions):
                    # Convert prediction to label ID (default to -1 if invalid)
                    pred_label = reverse_map.get(completion, -1)
                    predictions.append(pred_label)
                    ground_truths.append(example["label"])
                
                # Calculate macro-F1 score
                # Filter out invalid predictions (-1) for fair evaluation
                valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
                if len(valid_indices) > 0:
                    valid_preds = [predictions[i] for i in valid_indices]
                    valid_truths = [ground_truths[i] for i in valid_indices]
                    f1 = sklearn_f1_score(valid_truths, valid_preds, average='macro', zero_division=0.0)
                    rewards.append(f1)
                else:
                    # All predictions are invalid
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards
    return f1_classification_reward