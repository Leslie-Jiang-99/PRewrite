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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import argparse
from datetime import timedelta
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# 🔥🔥🔥 添加这段 - 强制设置 timeout
import torch.distributed as dist
_original_init = dist.init_process_group
def _wrapped_init(*args, **kwargs):
    if 'timeout' not in kwargs:
        kwargs['timeout'] = timedelta(seconds=7200)
        print(f"🔧 Setting torch.distributed timeout to 7200 seconds")
    else:
        print(f"⚠️ Original torch.distributed timeout is {kwargs['timeout']}")
        kwargs['timeout'] = timedelta(seconds=7200)
        print(f"🔧 Rewrite torch.distributed timeout to 7200 seconds")
    return _original_init(*args, **kwargs)
dist.init_process_group = _wrapped_init
# 🔥🔥🔥 monkey patch 结束

from accelerate import InitProcessGroupKwargs, logging
from datasets import load_dataset, load_from_disk

from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)

# patch the GRPOTrainer to use parallel reward computation
# from parallel_reward_patch import _calculate_rewards_parallel
# GRPOTrainer._calculate_rewards_serial = GRPOTrainer._calculate_rewards
# GRPOTrainer._calculate_rewards = _calculate_rewards_parallel
# print("✅ GRPOTrainer patched with parallel reward computation!")

from transformers import AutoTokenizer

from prewrite import (
    get_prewrite,
    get_accuracy_math,
    get_accuracy_classification,
    get_accuracy_nq_open_instruction_prewrite,
    get_f1_nq_open_instruction_prewrite,
    get_f1_classification,
    get_template_accuracy_math,
    get_accuracy_math_split,
)

from accelerate import PartialState

import asyncio

import sglang as sgl
import sglang.test.doc_patch
from sglang.utils import async_stream_and_merge, stream_and_merge

logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


reward_funcs_registry = {
    "prewrite_accuracy_reward": get_prewrite,
    "accuracy_math_reward": get_accuracy_math,
    "accuracy_classification_reward": get_accuracy_classification,
    "accuracy_nq_open_instruction_prewrite_reward": get_accuracy_nq_open_instruction_prewrite,
    "f1_nq_open_instruction_prewrite_reward": get_f1_nq_open_instruction_prewrite,
    "f1_classification_reward": get_f1_classification,
    "template_accuracy_math_reward": get_template_accuracy_math,
    "accuracy_math_reward_split": get_accuracy_math_split,
}

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

class ClassificationLogitProcessor(CustomLogitProcessor):
    """A classification logit processor that changes the logits to always
    sample the given token id.
    """

    def __call__(self, logits, custom_param_list):
        # Check that the number of logits matches the number of custom parameters
        assert logits.shape[0] == len(custom_param_list)
        key = "token_ids"

        for i, param_dict in enumerate(custom_param_list):
            # Mask all other tokens
            temp_logits = logits[i, :].clone()
            logits[i, :] = -float("inf")
            logits[i, param_dict[key]] = temp_logits[param_dict[key]]
        return logits

@dataclass
class PRewriteScriptArguments(ScriptArguments):
    """
    Script arguments for the PRewrite training script.

    Args:
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported values are:

                - `"prewrite_accuracy_reward"`
                - `"accuracy_math_reward"`
                - `"accuracy_classification_reward"`
                - `"accuracy_nq_open_instruction_prewrite_reward"`
                - `"f1_nq_open_instruction_prewrite_reward"`
                - `"f1_classification_reward"`
                - `"template_accuracy_math_reward"`
                - `"accuracy_math_reward_split"`
        generate_model_name_or_path (`str`, *optional*):
            Name or path of the generate model.
        generate_model_ip (`str`, *optional*):
            IP address of the generate model.
        generate_model_port (`int`, *optional*):
            Port of the generate model.
        tokenizer (`transformers.AutoTokenizer`, *optional*):
            Tokenizer for the generate model.
        generate_system_prompt (`str`, *optional*):
            System prompt for the generate model.
        test_size (`int`, *optional*):
            Size of the test set for the instruction prewrite accuracy reward.
        generate_dataset_seed (`int`, *optional*):
            Seed for the random number selection for the generate dataset.
        generate_is_test (`bool`, *optional*):
            Whether to use the test set for the instruction prewrite accuracy reward.
    """

    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: 'prewrite_accuracy_reward','accuracy_math_reward','accuracy_classification_reward','accuracy_nq_open_instruction_prewrite_reward','f1_nq_open_instruction_prewrite_reward','f1_classification_reward','template_accuracy_math_reward','accuracy_math_reward_split'"
        },
    )
    meta_instruction: str = field(
        default="Rewrite the following instruction via rephrasing and/or adding specific requirements. Add instructions which would be helpful to solve the problem correctly. Output the new instruction only.",
        metadata={"help": "Meta instruction for the rewrite model."},
    ) 
    generate_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name or path of the generate model."},
    )
    generate_model_ip: str = field(
        default="127.0.0.1",
        metadata={"help": "IP address of the generate model."},
    )
    generate_model_port: int = field(
        default=8000,
        metadata={"help": "Port of the generate model."},
    )
    generate_is_test: bool = field(
        default=False,
        metadata={"help": "Whether to use the test set for the instruction prewrite accuracy reward."},
    )
    tokenizer: AutoTokenizer = field(
        default=None,
        metadata={"help": "Tokenizer for the generate model."},
    )
    generate_system_prompt: str = field(
        default="",
        metadata={"help": "System prompt for the generate model."},
    )
    test_size: int = field(
        default=10,
        metadata={"help": "Size of the test set for the instruction prewrite accuracy reward."},
    )


def main(script_args, training_args, model_args, dataset_args):
    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                if func_name == "prewrite_accuracy_reward":
                    # Load the generate model tokenizer
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    generate_model_config = {
                        "temperature": script_args.generate_model_temperature,
                        "max_new_tokens": script_args.generate_model_max_new_tokens,
                    }
                    reward_funcs.append(get_prewrite(
                        generate_model_ip=script_args.generate_model_ip,
                        generate_model_port=script_args.generate_model_port,
                        generate_model_config=generate_model_config,
                        tokenizer=tokenizer,
                        system_prompt=script_args.generate_system_prompt,
                    ))
                elif func_name == "accuracy_math_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_accuracy_math(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            is_test=script_args.generate_is_test,
                        )
                    )
                elif func_name == "accuracy_classification_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_accuracy_classification(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            custom_logit_processor=ClassificationLogitProcessor(),
                            is_test=script_args.generate_is_test,
                        )
                    )
                elif func_name == "accuracy_nq_open_instruction_prewrite_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_accuracy_nq_open_instruction_prewrite(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            is_test=script_args.generate_is_test,
                        )
                    )
                elif func_name == "f1_nq_open_instruction_prewrite_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_f1_nq_open_instruction_prewrite(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            is_test=script_args.generate_is_test,
                        )
                    )
                elif func_name == "f1_classification_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_f1_classification(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            custom_logit_processor=ClassificationLogitProcessor(),
                            is_test=script_args.generate_is_test,
                        )
                    )
                elif func_name == "template_accuracy_math_reward":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_template_accuracy_math(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            is_test=script_args.generate_is_test,
                            print_text=False,
                        )
                    )
                elif func_name == "accuracy_math_reward_split":
                    if script_args.generate_model_name_or_path:
                        tokenizer = AutoTokenizer.from_pretrained(script_args.generate_model_name_or_path)
                    elif script_args.tokenizer:
                        tokenizer = script_args.tokenizer
                    else:
                        raise ValueError("Either `generate_model_name_or_path` or `tokenizer` must be provided.")
                    reward_funcs.append(
                        get_accuracy_math_split(
                            generate_model_ip=script_args.generate_model_ip,
                            generate_model_port=script_args.generate_model_port,
                            tokenizer=tokenizer,
                            system_prompt=script_args.generate_system_prompt,
                            test_size=script_args.test_size,
                            is_test=script_args.generate_is_test,
                            print_text=False,
                        )
                    )
                else:
                    reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    # Load the dataset
    dataset = load_from_disk(script_args.dataset_name)

    def add_meta_instruction(element):
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        element["prompt"] = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"{script_args.meta_instruction}\nInstruction: {element['instruction']}"
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return element

    dataset = dataset.map(add_meta_instruction, num_proc = 8, desc = "Adding meta instruction")

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (PRewriteScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("prewrite", help="Run the PRewrite training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
