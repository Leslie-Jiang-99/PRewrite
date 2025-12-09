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
import gc
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from typing import Optional
from tqdm import tqdm
import torch

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
from trl.models import unwrap_model_for_generation

# patch the GRPOTrainer to use parallel reward computation
# from parallel_reward_patch import _calculate_rewards_parallel
# GRPOTrainer._calculate_rewards_serial = GRPOTrainer._calculate_rewards
# GRPOTrainer._calculate_rewards = _calculate_rewards_parallel
# print("✅ GRPOTrainer patched with parallel reward computation!")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import transformers

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
        generate_model_offline (`bool`, *optional*):
            Whether to use the offline generate model.
        num_stage_steps (`int`, *optional*):
            Number of steps per stage. Model will be reset to base and dataset will be regenerated every num_stage_steps.
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
    generate_model_offline: bool = field(
        default=False,
        metadata={"help": "Whether to use the offline generate model."},
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
    num_stage_steps: int = field(
        default=100,
        metadata={"help": "Number of steps per stage. Model will be reset to base and dataset will be regenerated every num_stage_steps."},
    )


class MultiStageResetCallback(TrainerCallback):
    """
    Callback to reset model to base weights and regenerate dataset every num_stage_steps.
    """
    
    def __init__(
        self,
        num_stage_steps: int,
        base_model_path: str,
        model_init_kwargs: dict,
        script_args: PRewriteScriptArguments,
        model_args: ModelConfig,
        training_args: TrainingArguments,
        dataset,
        trainer=None,  # 将在创建 trainer 后设置
    ):
        self.num_stage_steps = num_stage_steps
        self.base_model_path = base_model_path
        self.model_init_kwargs = model_init_kwargs
        self.script_args = script_args
        self.model_args = model_args
        self.training_args = training_args
        self.dataset = dataset
        self.trainer = trainer  # 存储 trainer 引用
        self.last_reset_step = -1
        
        # 处理dtype
        dtype = self.model_init_kwargs.get("dtype")
        if isinstance(dtype, str) and dtype not in ["auto", "bfloat16", "float16", "float32"]:
            dtype = getattr(torch, dtype)
            self.model_init_kwargs["dtype"] = dtype
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Called at the end of a training step."""
        # 检查是否到了需要重置的步数
        if state.global_step <= 0:
            return
        
        # 检查是否是 num_stage_steps 的倍数，且不是刚刚重置过的步数
        if (state.global_step % self.num_stage_steps == 0 and 
            state.global_step != self.last_reset_step):
            
            # 尝试从 kwargs 获取 trainer，如果没有则使用 self.trainer
            trainer = kwargs.get("trainer") or self.trainer
            if trainer is None:
                # 如果还是没有，尝试从 kwargs 获取其他方式
                # 在某些情况下，trainer 可能不在 kwargs 中，我们需要在创建 trainer 后设置它
                return
            
            self.last_reset_step = state.global_step
            stage_num = state.global_step // self.num_stage_steps
            
            trainer.accelerator.print(f"🔄 Stage {stage_num}: Resetting model and regenerating dataset at step {state.global_step}...")
            
            try:
                # 1. 重新生成数据集
                self._regenerate_dataset(trainer)

                # 2. 重置模型权重到 base 模型
                self._reset_model_weights(trainer)
                
                # 3. 更新 trainer 的数据集
                trainer.train_dataset = self.dataset[self.script_args.dataset_train_split]
                if self.training_args.eval_strategy != "no":
                    trainer.eval_dataset = self.dataset[self.script_args.dataset_test_split]

                trainer.callback_handler.train_dataloader = trainer.get_train_dataloader()
                
                trainer.accelerator.print(f"✅ All dataloader caches cleared and updated")
                
                trainer.accelerator.print(f"✅ Stage {stage_num}: Model reset and dataset regenerated successfully.")

            except Exception as e:
                trainer.accelerator.print(f"⚠️  Error during stage reset at step {state.global_step}: {e}")
                import traceback
                trainer.accelerator.print(f"   Traceback: {traceback.format_exc()}")
    
    def _reset_model_weights(self, trainer):
        """Reset model weights to base model."""
        trainer.accelerator.print(f"📥 Loading base model weights from {self.base_model_path}...")
        
        # 在CPU上加载base模型的state_dict
        config = AutoConfig.from_pretrained(self.base_model_path)
        architecture = getattr(transformers, config.architectures[0])
        base_model = architecture.from_pretrained(self.base_model_path, **self.model_init_kwargs)
        base_state_dict = base_model.state_dict()
        del base_model  # 释放CPU内存
        gc.collect()
        
        # 获取实际的模型（unwrap DDP/DeepSpeed wrapper）
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # 检查是否是 PEFT 模型
        from peft import PeftModel
        is_peft = isinstance(unwrapped_model, PeftModel)
        
        # 在 DeepSpeed ZeRO-3 下，参数被分片到不同 GPU
        is_deepspeed_zero3 = (
            trainer.accelerator.state.deepspeed_plugin is not None 
            and trainer.accelerator.state.deepspeed_plugin.zero_stage == 3
        )
        
        # 过滤掉形状为空的参数
        filtered_base_state_dict = {}
        empty_shape_keys = []
        for key, value in base_state_dict.items():
            if value.numel() > 0:
                # 如果是 PEFT 模型，过滤掉 adapter 相关的键
                if is_peft and any(adapter_key in key for adapter_key in ['lora_', 'adapter_', 'prompt_']):
                    continue
                filtered_base_state_dict[key] = value
            else:
                empty_shape_keys.append(key)
        
        if empty_shape_keys:
            trainer.accelerator.print(f"⚠️  Skipped {len(empty_shape_keys)} keys with empty shape in base_state_dict")
        
        if is_deepspeed_zero3:
            # 在 ZeRO-3 下，直接使用 load_state_dict，DeepSpeed 会自动处理参数分片
            trainer.accelerator.print(f"📦 DeepSpeed ZeRO-3 detected: Loading weights with automatic sharding...")
            
            # 确定要加载的模型（如果是 PEFT，加载到 base_model）
            target_model = unwrapped_model.get_base_model() if is_peft else unwrapped_model
            
            # 在 DeepSpeed ZeRO-3 下，直接调用 load_state_dict
            missing_keys, unexpected_keys = target_model.load_state_dict(filtered_base_state_dict, strict=False)
            trainer.accelerator.print(f"✅ Base model weights loaded (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
            
            # 同步所有进程
            trainer.accelerator.wait_for_everyone()
        else:
            # 非 ZeRO-3 环境，直接比较和加载
            target_model = unwrapped_model.get_base_model() if is_peft else unwrapped_model
            current_state_dict = target_model.state_dict()
            
            # 再次过滤，确保形状匹配
            final_filtered_dict = {}
            skipped_keys = []
            for key, value in filtered_base_state_dict.items():
                if key in current_state_dict:
                    # 检查形状是否匹配
                    if value.shape == current_state_dict[key].shape:
                        final_filtered_dict[key] = value
                    else:
                        skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {current_state_dict[key].shape})")
                else:
                    skipped_keys.append(f"{key} (not in current model)")
            
            if skipped_keys:
                trainer.accelerator.print(f"⚠️  Skipped {len(skipped_keys)} keys when loading base model weights:")
                for key in skipped_keys[:10]:  # 只打印前10个
                    trainer.accelerator.print(f"  - {key}")
                if len(skipped_keys) > 10:
                    trainer.accelerator.print(f"  ... and {len(skipped_keys) - 10} more")
            
            missing_keys, unexpected_keys = target_model.load_state_dict(final_filtered_dict, strict=False)
            trainer.accelerator.print(f"✅ Base model weights loaded (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
        
        # 清理
        del filtered_base_state_dict
        del base_state_dict
        gc.collect()
    
    def _regenerate_dataset(self, trainer):
        """Regenerate dataset by changing initial instructions."""
        # 获取实际的模型（unwrap DDP/DeepSpeed wrapper）
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        
        # 在函数外部创建 tokenizer，避免在 map 中重复创建
        tokenizer_for_generation = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        
        def change_initial_instruction(element):
            inputs = tokenizer_for_generation([element["prompt"]], return_tensors="pt")
            # 使用accelerator.device而不是trainer.model.device（DeepSpeed包装的模型可能没有device属性）
            device = trainer.accelerator.device if hasattr(trainer.accelerator, 'device') else next(unwrapped_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 对于DeepSpeed ZeRO-3，设置 gather_deepspeed3_params=False 来避免 OOM
            gather_deepspeed3_params = False  # 强制设置为 False 以避免 OOM
            with (
                unwrap_model_for_generation(
                    trainer.model, trainer.accelerator, gather_deepspeed3_params=gather_deepspeed3_params
                ) as model_for_generation,
                torch.no_grad(),
            ):
                output_ids = model_for_generation.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.training_args.max_completion_length,
                    do_sample=False
                )
            
            # 解码输出
            # 注意：根据实际模型输出格式调整分隔符（可能是 </think>）
            decoded = tokenizer_for_generation.decode(output_ids[0], skip_special_tokens=True)
            if "</think>" in decoded:
                element["instruction"] = decoded.split("</think>")[-1].strip()
            else:
                element["instruction"] = decoded.strip()
            
            trainer.accelerator.print(f"✅ Instruction: {element['instruction']}")

            # 立即清理输入和输出张量
            del inputs, output_ids
            
            element["prompt"] = tokenizer_for_generation.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": f"{self.script_args.meta_instruction}\nInstruction: {element['instruction']}"
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return element
        
        trainer.accelerator.print(f"🔄 Processing dataset (using gather_deepspeed3_params=False to avoid OOM)...")
        self.dataset = self.dataset.map(change_initial_instruction, desc="Changing initial instruction")
        trainer.accelerator.print(f"✅ Dataset changed.")
        
        # 清理 tokenizer 和生成过程中的显存
        del tokenizer_for_generation
        gc.collect()
        
        # 清理 GPU 显存
        if torch.cuda.is_available():
            try:
                local_rank = trainer.accelerator.local_process_index if hasattr(trainer.accelerator, 'local_process_index') else 0
                device_id = local_rank % torch.cuda.device_count()
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as e:
                trainer.accelerator.print(f"⚠️  Warning: Failed to clear GPU {device_id} cache after dataset processing: {e}")
        gc.collect()


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
        generate_model_offline (`bool`, *optional*):
            Whether to use the offline generate model.
        num_stages (`int`, *optional*):
            Number of stages to train the model.
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
    generate_model_offline: bool = field(
        default=False,
        metadata={"help": "Whether to use the offline generate model."},
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
    num_stages: int = field(
        default=10,
        metadata={"help": "Number of stages to train the model."},
    )
    num_stage_steps: int = field(
        default=100,
        metadata={"help": "Number of steps per stage. Model will be reset to base and dataset will be regenerated every num_stage_steps."},
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
                            is_test=False,
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
                            is_test=False,
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
                            is_test=False,
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
                            is_test=False,
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
                            is_test=False,
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
                            is_test=False,
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
                            is_test=True,
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

    dataset = dataset.map(add_meta_instruction, desc = "Adding meta instruction")

    # 准备 callback 参数
    base_model_path = model_args.model_name_or_path
    model_init_kwargs = training_args.model_init_kwargs or {}
    
    # 创建 multi-stage reset callback（先不传入 trainer）
    multi_stage_callback = MultiStageResetCallback(
        num_stage_steps=script_args.num_stage_steps,
        base_model_path=base_model_path,
        model_init_kwargs=model_init_kwargs,
        script_args=script_args,
        model_args=model_args,
        training_args=training_args,
        dataset=dataset,
        trainer=None,  # 稍后设置
    )
    
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=[multi_stage_callback],  # 添加 callback
    )
    
    # 设置 callback 的 trainer 引用
    multi_stage_callback.trainer = trainer
    
    # 开始训练（callback 会在每个 num_stage_steps 步后自动重置模型和数据集）
    trainer.train()
    
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
