from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

import sys
sys.path.insert(0, "/root/workspace/PRewrite")

from prewrite import (
    get_accuracy_math,
    get_accuracy_classification,
    get_accuracy_nq_open_instruction_prewrite,
    get_f1_nq_open_instruction_prewrite,
    get_f1_classification,
)
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

generate_model_ip = "10.43.2.16"
generate_model_port = "33337"
generate_model_tokenizer = AutoTokenizer.from_pretrained("/root/group-shared/jrc/base_models/Qwen3-32B")

reward_funcs = [
    get_accuracy_math(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, is_test=True),
    get_accuracy_classification(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, ClassificationLogitProcessor(), is_test=True),
    get_accuracy_nq_open_instruction_prewrite(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, is_test=True),
    get_f1_nq_open_instruction_prewrite(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, is_test=True),
    get_f1_classification(generate_model_ip, generate_model_port, generate_model_tokenizer, "", 100, ClassificationLogitProcessor(), is_test=True),
]

extended_dataset = load_from_disk("/root/workspace/PRewrite/data/extended_dataset")

next_stage_extended_dataset = load_from_disk("/root/workspace/PRewrite/data/next_stage_extended_dataset")

dataset = next_stage_extended_dataset["train"]

model_name_or_path = "your_model_path"

rewrite_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=True
)

meta_instruction = "Rewrite the following instruction via rephrasing and/or adding specific requirements. Add instructions which would be helpful to solve the problem correctly. Output the new instruction only."

f = open("your_output_file.md", "a")
f.write(f"# Model: {model_name_or_path}\n\n")
for example in dataset:
    f.write(f"## dataset_name: {example['dataset_name']}\n\n")
    initial_instruction = example["instruction"]
    print(f"initial_instruction: {initial_instruction}")
    f.write(f"initial_instruction: {initial_instruction}\n")
    for reward_func in reward_funcs:
        reward = reward_func(completions=[initial_instruction], dataset_name = [example["dataset_name"]])
        print(f"{reward_func.__name__}: {reward}")
        f.write(f"- {reward_func.__name__}: {reward}\n")
    f.write("\n")
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{meta_instruction}\nInstruction: {initial_instruction}"
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(rewrite_model.device)
    output_ids = rewrite_model.generate(
        input_ids=inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1024,
        do_sample=False,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    rewrite_instruction = output_text.split("</think>\n")[-1]
    print(f"rewrite_instruction: {rewrite_instruction}")
    f.write(f"rewrite_instruction: {rewrite_instruction}\n")
    for reward_func in reward_funcs:
        reward = reward_func(completions = [rewrite_instruction], dataset_name = [example["dataset_name"]])
        print(f"{reward_func.__name__}: {reward}")
        f.write(f"- {reward_func.__name__}: {reward}\n")
f.write("\n")
f.close()