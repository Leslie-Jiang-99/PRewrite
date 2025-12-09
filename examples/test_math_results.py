from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, "/root/workspace/PRewrite")

from prewrite import (
    get_accuracy_math,
)

generate_model_ip = "0.0.0.0"
generate_model_port = "33337"
generate_model_tokenizer = AutoTokenizer.from_pretrained("/root/group-shared/jrc/base_models/Qwen3-32B")

initial_instruction = "Answer the hard math problem step by step, and provide a boxed final answer at the end."
rewrite_instruction = r"""Provide a detailed, step-by-step solution to the given hard math problem, including all necessary mathematical steps, thorough explanations for each step, and logical justifications for your reasoning. Use proper mathematical notation throughout, clearly label each part of the solution (e.g., "Step 1: Problem Analysis," "Step 2: Strategy and Approach," etc.), and ensure the solution flows logically from beginning to end. After arriving at the final answer, present it in a box at the end using the format $\boxed{\text{answer}}$. Additionally, check your work for any possible errors, and verify your solution by applying an alternative method or testing it with a concrete example if applicable."""

initial_rewards = []
rewrite_rewards = []

reward_func = get_accuracy_math(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, is_test=True, print_text=False)

for i in tqdm(range(64)):
    # initial_reward = reward_func(completions = [initial_instruction], dataset_name = ["aime_2024"])[0]
    rewrite_reward = reward_func(completions = [rewrite_instruction], dataset_name = ["MATH-500"])[0]
    # initial_rewards.append(initial_reward)
    rewrite_rewards.append(rewrite_reward)
    # print(f"mean initial_reward: {np.mean(initial_rewards)}")
    print(f"mean rewrite_reward: {np.mean(rewrite_rewards)}")
    if i > 0:
        # print(f"std initial_reward: {np.std(initial_rewards)}")
        print(f"std rewrite_reward: {np.std(rewrite_rewards)}")
    # print("initial_rewards: ", initial_rewards)
    print("rewrite_rewards: ", rewrite_rewards)
    sys.stdout.flush()             

