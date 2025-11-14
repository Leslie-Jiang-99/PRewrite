"""
并行优化的reward functions

相比原版的优化：
1. 使用ThreadPoolExecutor并行处理多个completions
2. 使用requests.Session复用连接
3. 批量API调用（如果服务器支持）
"""

import time
import random
import requests
import numpy as np
from typing import Callable, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from math_verify import LatexExtractionConfig, NormalizationConfig, parse, verify


def get_accuracy_math_parallel(
    generate_model_ip: str, 
    generate_model_port: int, 
    tokenizer: AutoTokenizer, 
    system_prompt: str = "", 
    test_size: int = 10, 
    is_test: bool = False,
    max_workers: int = 8  # 🔥 新增：并行度
) -> Callable:
    """
    并行优化版本的accuracy_math_reward
    
    优化点：
    1. 使用ThreadPoolExecutor并行处理多个completions
    2. 复用requests.Session
    3. 每个completion的处理互不阻塞
    """
    
    # 创建共享的session（线程安全）
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_completion(
        instruction, 
        single_dataset_name, 
        train_dataset_dict, 
        test_dataset_dict
    ):
        """处理单个completion的reward计算"""
        try:
            if single_dataset_name not in ["gsm8k", "MATH-500"]:
                return 0.0
            
            # 选择数据集
            if is_test:
                dataset = test_dataset_dict[single_dataset_name]
                sample_size = len(dataset)
            else:
                dataset = train_dataset_dict[single_dataset_name]
                sample_size = min(test_size, len(dataset))
            
            # 采样和准备prompts
            scores = []
            generate_model_config = {
                "temperature": 0.0,
                "max_new_tokens": 2048,
            }
            random.seed(time.time() + random.random())  # 避免多线程冲突
            indices = random.sample(range(len(dataset)), sample_size)
            
            prompts = []
            from prewrite.prewrite_rewards import get_messages  # 导入辅助函数
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt)
            
            # 发送API请求
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompts,
                    "sampling_params": generate_model_config,
                },
                timeout=60  # 添加超时
            )
            
            if response.status_code != 200:
                return None
            
            completions = [output["text"] for output in response.json()]
            
            # 验证答案
            for example, completion in zip(dataset.select(indices), completions):
                gold_parsed = parse(
                    example["answer"],
                    extraction_mode="first_match",
                )
                
                if len(gold_parsed) != 0:
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
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    
                    try:
                        score = float(verify(gold_parsed, answer_parsed))
                    except Exception:
                        score = None
                else:
                    score = float(completion.strip().lower() == example["answer"].strip().lower())
                
                if score is not None:
                    scores.append(score)
            
            return np.mean(scores) if scores else None
            
        except Exception as e:
            print(f"❌ Error processing completion: {e}")
            return None
    
    def accuracy_math_reward_parallel(
        completions: List[List[dict[str, str]]], 
        dataset_name: List[str], 
        **kwargs
    ) -> List[Optional[float]]:
        """
        并行计算所有completions的rewards
        """
        from datasets import load_dataset
        
        # 加载数据集（只加载一次）
        train_dataset_dict = {
            "gsm8k": load_dataset("openai/gsm8k", "main", split="train"),
            "MATH-500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        }
        test_dataset_dict = {
            "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
            "MATH-500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        }
        
        n_completions = len(completions)
        
        # 如果只有1个completion，直接处理
        if n_completions == 1:
            reward = process_single_completion(
                completions[0], 
                dataset_name[0],
                train_dataset_dict,
                test_dataset_dict
            )
            return [reward]
        
        # 🔥 并行处理多个completions
        print(f"🔄 [Batch Parallel] Processing {n_completions} completions with {max_workers} workers")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    process_single_completion,
                    completions[idx],
                    dataset_name[idx],
                    train_dataset_dict,
                    test_dataset_dict
                ): idx
                for idx in range(n_completions)
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    reward = future.result()
                    rewards[idx] = reward
                except Exception as e:
                    print(f"❌ Failed to get reward for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    # 🔥 标记为支持batch并行
    accuracy_math_reward_parallel.__parallel_batch__ = True
    
    return accuracy_math_reward_parallel


# ============================================================================
# 更激进的优化：批量API调用
# ============================================================================

def get_accuracy_math_batch_api(
    generate_model_ip: str, 
    generate_model_port: int, 
    tokenizer: AutoTokenizer, 
    system_prompt: str = "", 
    test_size: int = 10, 
    is_test: bool = False,
) -> Callable:
    """
    最激进的优化：一次API调用处理所有completions
    
    假设：你的generate model API支持batch处理
    """
    
    session = requests.Session()
    
    def accuracy_math_reward_batch(
        completions: List[List[dict[str, str]]], 
        dataset_name: List[str], 
        **kwargs
    ) -> List[Optional[float]]:
        from datasets import load_dataset
        
        train_dataset_dict = {
            "gsm8k": load_dataset("openai/gsm8k", "main", split="train"),
            "MATH-500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        }
        test_dataset_dict = {
            "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
            "MATH-500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
        }
        
        rewards = []
        
        # 🔥 收集所有prompts到一个大batch
        all_prompts = []
        completion_to_prompts_range = []  # 记录每个completion对应的prompt范围
        
        from prewrite.prewrite_rewards import get_messages
        
        for instruction, single_dataset_name in zip(completions, dataset_name):
            if single_dataset_name not in ["gsm8k", "MATH-500"]:
                completion_to_prompts_range.append((len(all_prompts), len(all_prompts)))
                continue
            
            dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
            sample_size = len(dataset) if is_test else min(test_size, len(dataset))
            
            indices = random.sample(range(len(dataset)), sample_size)
            
            start_idx = len(all_prompts)
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                all_prompts.append(prompt)
            
            end_idx = len(all_prompts)
            completion_to_prompts_range.append((start_idx, end_idx))
        
        # 🔥 一次API调用处理所有prompts
        print(f"🚀 [Batch API] Sending {len(all_prompts)} prompts in 1 API call")
        
        try:
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": all_prompts,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 2048,
                    },
                },
                timeout=180  # 更长的超时
            )
            
            if response.status_code != 200:
                return [None] * len(completions)
            
            all_completions = [output["text"] for output in response.json()]
            
            # 计算每个completion的reward
            for idx, (instruction, single_dataset_name) in enumerate(zip(completions, dataset_name)):
                start_idx, end_idx = completion_to_prompts_range[idx]
                
                if start_idx == end_idx:
                    rewards.append(0.0)
                    continue
                
                dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
                sample_size = end_idx - start_idx
                
                scores = []
                for j, completion_text in enumerate(all_completions[start_idx:end_idx]):
                    example = dataset[j]
                    
                    gold_parsed = parse(example["answer"], extraction_mode="first_match")
                    if len(gold_parsed) != 0:
                        answer_parsed = parse(
                            completion_text,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False, malformed_operators=False,
                                        basic_latex=True, boxed="all", units=True,
                                    ),
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        
                        try:
                            score = float(verify(gold_parsed, answer_parsed))
                        except Exception:
                            score = None
                    else:
                        score = float(completion_text.strip().lower() == example["answer"].strip().lower())
                    
                    if score is not None:
                        scores.append(score)
                
                rewards.append(np.mean(scores) if scores else None)
        
        except Exception as e:
            print(f"❌ Batch API call failed: {e}")
            return [None] * len(completions)
        
        return rewards
    
    accuracy_math_reward_batch.__parallel_batch__ = True
    
    return accuracy_math_reward_batch


# ============================================================================
# 并行版本：get_prewrite
# ============================================================================

def get_prewrite_parallel(
    generate_model_ip: str,
    generate_model_port: int,
    generate_model_config: dict,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
    max_workers: int = 8
) -> Callable:
    """并行优化版本的prewrite_accuracy_reward"""
    
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_prewrite(completion, answer):
        """处理单个completion"""
        try:
            from trl.import_utils import is_math_verify_available
            if not is_math_verify_available():
                raise ImportError("Please install the `math_verify` package")
            
            # 生成prompt
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, 
                 {"role": "user", "content": completion}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # API调用
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompt,
                    "sampling_params": generate_model_config,
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return None
            
            content = response.json()["text"]
            
            # 验证答案
            gold_parsed = parse(answer, extraction_mode="first_match")
            
            if len(gold_parsed) != 0:
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
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = None
            else:
                reward = float(content.strip().lower() == answer.strip().lower())
            
            return reward
            
        except Exception as e:
            print(f"❌ Error in prewrite: {e}")
            return None
    
    def prewrite_accuracy_reward_parallel(
        completions: List[List[dict[str, str]]],
        answer: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        """并行计算prewrite rewards"""
        
        n_completions = len(completions)
        
        if n_completions == 1:
            reward = process_single_prewrite(completions[0], answer[0])
            return [reward]
        
        print(f"🔄 [Prewrite Parallel] Processing {n_completions} completions")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            future_to_idx = {
                executor.submit(process_single_prewrite, completions[idx], answer[idx]): idx
                for idx in range(n_completions)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    print(f"❌ Failed for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    prewrite_accuracy_reward_parallel.__parallel_batch__ = True
    return prewrite_accuracy_reward_parallel


# ============================================================================
# 并行版本：get_accuracy_classification
# ============================================================================

def get_accuracy_classification_parallel(
    generate_model_ip: str,
    generate_model_port: int,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
    test_size: int = 10,
    custom_logit_processor = None,
    is_test: bool = False,
    max_workers: int = 8
) -> Callable:
    """并行优化版本的accuracy_classification_reward"""
    
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_classification(
        instruction,
        single_dataset_name,
        train_dataset_dict,
        test_dataset_dict,
        task_classification_map,
        task_token_ids
    ):
        """处理单个classification completion"""
        try:
            if single_dataset_name not in ["ag_news", "sst2", "copa", "boolq", "cb"]:
                return 0.0
            
            dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
            sample_size = len(dataset) if is_test else min(test_size, len(dataset))
            
            generate_model_config = {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "custom_params": {"token_ids": task_token_ids[single_dataset_name]},
            }
            
            classification_map = task_classification_map[single_dataset_name]
            
            random.seed(time.time() + random.random())
            indices = random.sample(range(len(dataset)), sample_size)
            
            prompts = []
            from prewrite.prewrite_rewards import get_messages
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt)
            
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompts,
                    "sampling_params": generate_model_config,
                    "custom_logit_processor": custom_logit_processor.to_str() if custom_logit_processor else None,
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return None
            
            completions = [output["text"] for output in response.json()]
            
            scores = []
            for example, completion in zip(dataset.select(indices), completions):
                if completion == classification_map[example["label"]]:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            
            return np.mean(scores) if scores else None
            
        except Exception as e:
            print(f"❌ Error in classification: {e}")
            return None
    
    def accuracy_classification_reward_parallel(
        completions: List[List[dict[str, str]]],
        dataset_name: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        """并行计算classification rewards"""
        from datasets import load_dataset
        
        # 加载数据集（使用本地路径）
        from prewrite.prewrite_rewards import (
            ag_news_dataset_path, ag_news_dataset_config,
            sst2_dataset_path, sst2_dataset_config,
            copa_dataset_path, copa_dataset_config,
            boolq_dataset_path, boolq_dataset_config,
            cb_dataset_path, cb_dataset_config,
            task_classification_map, task_token_ids
        )
        
        train_dataset_dict = {
            "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config, split="train"),
            "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config, split="train"),
            "copa": load_dataset(copa_dataset_path, copa_dataset_config, split="train"),
            "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config, split="train"),
            "cb": load_dataset(cb_dataset_path, cb_dataset_config, split="train"),
        }
        test_dataset_dict = {
            "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config, split="test"),
            "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config, split="validation"),
            "copa": load_dataset(copa_dataset_path, copa_dataset_config, split="validation"),
            "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config, split="validation"),
            "cb": load_dataset(cb_dataset_path, cb_dataset_config, split="validation"),
        }
        
        n_completions = len(completions)
        
        if n_completions == 1:
            reward = process_single_classification(
                completions[0], dataset_name[0],
                train_dataset_dict, test_dataset_dict,
                task_classification_map, task_token_ids
            )
            return [reward]
        
        print(f"🔄 [Classification Parallel] Processing {n_completions} completions")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_classification,
                    completions[idx], dataset_name[idx],
                    train_dataset_dict, test_dataset_dict,
                    task_classification_map, task_token_ids
                ): idx
                for idx in range(n_completions)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    print(f"❌ Failed for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    accuracy_classification_reward_parallel.__parallel_batch__ = True
    return accuracy_classification_reward_parallel


# ============================================================================
# 并行版本：get_accuracy_nq_open_instruction_prewrite
# ============================================================================

def get_accuracy_nq_open_instruction_prewrite_parallel(
    generate_model_ip: str,
    generate_model_port: int,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
    test_size: int = 10,
    is_test: bool = False,
    max_workers: int = 8
) -> Callable:
    """并行优化版本的accuracy_nq_open"""
    
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_nq_accuracy(
        instruction,
        single_dataset_name,
        train_dataset_dict,
        test_dataset_dict
    ):
        """处理单个NQ Open accuracy"""
        try:
            if single_dataset_name != "nq_open":
                return 0.0
            
            dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
            sample_size = len(dataset) if is_test else min(test_size, len(dataset))
            
            generate_model_config = {
                "temperature": 0.0,
                "max_new_tokens": 100,
            }
            
            random.seed(time.time() + random.random())
            indices = random.sample(range(len(dataset)), sample_size)
            
            prompts = []
            from prewrite.prewrite_rewards import get_messages, normalize_answer
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt)
            
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompts,
                    "sampling_params": generate_model_config,
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return None
            
            completions = [output["text"] for output in response.json()]
            
            scores = []
            for example, completion in zip(dataset.select(indices), completions):
                flag = False
                for answer in example["answer"]:
                    if normalize_answer(completion) == normalize_answer(answer):
                        flag = True
                        break
                scores.append(1.0 if flag else 0.0)
            
            return np.mean(scores) if scores else None
            
        except Exception as e:
            print(f"❌ Error in NQ Open accuracy: {e}")
            return None
    
    def accuracy_nq_open_reward_parallel(
        completions: List[List[dict[str, str]]],
        dataset_name: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        """并行计算NQ Open accuracy rewards"""
        from datasets import load_dataset
        from prewrite.prewrite_rewards import nq_open_dataset_path, nq_open_dataset_config
        
        train_dataset_dict = {
            "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config, split="train"),
        }
        test_dataset_dict = {
            "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config, split="validation"),
        }
        
        n_completions = len(completions)
        
        if n_completions == 1:
            reward = process_single_nq_accuracy(
                completions[0], dataset_name[0],
                train_dataset_dict, test_dataset_dict
            )
            return [reward]
        
        print(f"🔄 [NQ Accuracy Parallel] Processing {n_completions} completions")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_nq_accuracy,
                    completions[idx], dataset_name[idx],
                    train_dataset_dict, test_dataset_dict
                ): idx
                for idx in range(n_completions)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    print(f"❌ Failed for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    accuracy_nq_open_reward_parallel.__parallel_batch__ = True
    return accuracy_nq_open_reward_parallel


# ============================================================================
# 并行版本：get_f1_nq_open_instruction_prewrite
# ============================================================================

def get_f1_nq_open_instruction_prewrite_parallel(
    generate_model_ip: str,
    generate_model_port: int,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
    test_size: int = 10,
    is_test: bool = False,
    max_workers: int = 8
) -> Callable:
    """并行优化版本的f1_nq_open"""
    
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_nq_f1(
        instruction,
        single_dataset_name,
        train_dataset_dict,
        test_dataset_dict
    ):
        """处理单个NQ Open F1"""
        try:
            if single_dataset_name != "nq_open":
                return 0.0
            
            dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
            sample_size = len(dataset) if is_test else min(test_size, len(dataset))
            
            generate_model_config = {
                "temperature": 0.0,
                "max_new_tokens": 100,
            }
            
            random.seed(time.time() + random.random())
            indices = random.sample(range(len(dataset)), sample_size)
            
            prompts = []
            from prewrite.prewrite_rewards import get_messages, f1_score
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt)
            
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompts,
                    "sampling_params": generate_model_config,
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return None
            
            completions = [output["text"] for output in response.json()]
            
            scores = []
            for example, completion in zip(dataset.select(indices), completions):
                score = 0
                for answer in example["answer"]:
                    score = max(score, f1_score(answer, completion))
                scores.append(score)
            
            return np.mean(scores) if scores else None
            
        except Exception as e:
            print(f"❌ Error in NQ Open F1: {e}")
            return None
    
    def f1_nq_open_reward_parallel(
        completions: List[List[dict[str, str]]],
        dataset_name: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        """并行计算NQ Open F1 rewards"""
        from datasets import load_dataset
        from prewrite.prewrite_rewards import nq_open_dataset_path, nq_open_dataset_config
        
        train_dataset_dict = {
            "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config, split="train"),
        }
        test_dataset_dict = {
            "nq_open": load_dataset(nq_open_dataset_path, nq_open_dataset_config, split="validation"),
        }
        
        n_completions = len(completions)
        
        if n_completions == 1:
            reward = process_single_nq_f1(
                completions[0], dataset_name[0],
                train_dataset_dict, test_dataset_dict
            )
            return [reward]
        
        print(f"🔄 [NQ F1 Parallel] Processing {n_completions} completions")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_nq_f1,
                    completions[idx], dataset_name[idx],
                    train_dataset_dict, test_dataset_dict
                ): idx
                for idx in range(n_completions)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    print(f"❌ Failed for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    f1_nq_open_reward_parallel.__parallel_batch__ = True
    return f1_nq_open_reward_parallel


# ============================================================================
# 并行版本：get_f1_classification
# ============================================================================

def get_f1_classification_parallel(
    generate_model_ip: str,
    generate_model_port: int,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
    test_size: int = 10,
    custom_logit_processor = None,
    is_test: bool = False,
    max_workers: int = 8
) -> Callable:
    """并行优化版本的f1_classification_reward"""
    
    session = requests.Session()
    session.mount('http://', requests.adapters.HTTPAdapter(
        pool_connections=max_workers,
        pool_maxsize=max_workers * 2,
        max_retries=3
    ))
    
    def process_single_f1_classification(
        instruction,
        single_dataset_name,
        train_dataset_dict,
        test_dataset_dict,
        task_classification_map,
        task_token_ids
    ):
        """处理单个F1 classification"""
        try:
            if single_dataset_name not in ["ag_news", "sst2", "copa", "boolq", "cb"]:
                return 0.0
            
            dataset = test_dataset_dict[single_dataset_name] if is_test else train_dataset_dict[single_dataset_name]
            sample_size = len(dataset) if is_test else min(test_size, len(dataset))
            
            generate_model_config = {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "custom_params": {"token_ids": task_token_ids[single_dataset_name]},
            }
            
            classification_map = task_classification_map[single_dataset_name]
            reverse_map = {v: k for k, v in classification_map.items()}
            
            random.seed(time.time() + random.random())
            indices = random.sample(range(len(dataset)), sample_size)
            
            prompts = []
            from prewrite.prewrite_rewards import get_messages
            for example in dataset.select(indices):
                prompt = tokenizer.apply_chat_template(
                    get_messages(instruction, example, single_dataset_name, system_prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts.append(prompt)
            
            response = session.post(
                f"http://{generate_model_ip}:{generate_model_port}/generate",
                json={
                    "text": prompts,
                    "sampling_params": generate_model_config,
                    "custom_logit_processor": custom_logit_processor.to_str() if custom_logit_processor else None,
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return None
            
            completions = [output["text"] for output in response.json()]
            
            predictions = []
            ground_truths = []
            
            for example, completion in zip(dataset.select(indices), completions):
                pred_label = reverse_map.get(completion, -1)
                predictions.append(pred_label)
                ground_truths.append(example["label"])
            
            # Calculate macro-F1
            from sklearn.metrics import f1_score as sklearn_f1_score
            valid_indices = [i for i, pred in enumerate(predictions) if pred != -1]
            
            if len(valid_indices) > 0:
                valid_preds = [predictions[i] for i in valid_indices]
                valid_truths = [ground_truths[i] for i in valid_indices]
                f1 = sklearn_f1_score(valid_truths, valid_preds, average='macro', zero_division=0.0)
                return f1
            else:
                return 0.0
            
        except Exception as e:
            print(f"❌ Error in F1 classification: {e}")
            return None
    
    def f1_classification_reward_parallel(
        completions: List[List[dict[str, str]]],
        dataset_name: List[str],
        **kwargs
    ) -> List[Optional[float]]:
        """并行计算F1 classification rewards"""
        from datasets import load_dataset
        
        from prewrite.prewrite_rewards import (
            ag_news_dataset_path, ag_news_dataset_config,
            sst2_dataset_path, sst2_dataset_config,
            copa_dataset_path, copa_dataset_config,
            boolq_dataset_path, boolq_dataset_config,
            cb_dataset_path, cb_dataset_config,
            task_classification_map, task_token_ids
        )
        
        train_dataset_dict = {
            "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config, split="train"),
            "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config, split="train"),
            "copa": load_dataset(copa_dataset_path, copa_dataset_config, split="train"),
            "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config, split="train"),
            "cb": load_dataset(cb_dataset_path, cb_dataset_config, split="train"),
        }
        test_dataset_dict = {
            "ag_news": load_dataset(ag_news_dataset_path, ag_news_dataset_config, split="test"),
            "sst2": load_dataset(sst2_dataset_path, sst2_dataset_config, split="validation"),
            "copa": load_dataset(copa_dataset_path, copa_dataset_config, split="validation"),
            "boolq": load_dataset(boolq_dataset_path, boolq_dataset_config, split="validation"),
            "cb": load_dataset(cb_dataset_path, cb_dataset_config, split="validation"),
        }
        
        n_completions = len(completions)
        
        if n_completions == 1:
            reward = process_single_f1_classification(
                completions[0], dataset_name[0],
                train_dataset_dict, test_dataset_dict,
                task_classification_map, task_token_ids
            )
            return [reward]
        
        print(f"🔄 [F1 Classification Parallel] Processing {n_completions} completions")
        
        rewards = [None] * n_completions
        
        with ThreadPoolExecutor(max_workers=min(max_workers, n_completions)) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_f1_classification,
                    completions[idx], dataset_name[idx],
                    train_dataset_dict, test_dataset_dict,
                    task_classification_map, task_token_ids
                ): idx
                for idx in range(n_completions)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    print(f"❌ Failed for completion {idx}: {e}")
                    rewards[idx] = None
        
        return rewards
    
    f1_classification_reward_parallel.__parallel_batch__ = True
    return f1_classification_reward_parallel

