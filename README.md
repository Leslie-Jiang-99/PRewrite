# PRewrite

使用强化学习（RL）自动重写和优化指令（Prompt）的项目。通过 GRPO（Group Relative Policy Optimization）算法训练模型，使其能够自动重写原始指令，通过改写和添加特定要求来提升下游任务的性能。

## 项目简介

PRewrite 是一个基于强化学习的指令重写系统，核心思想是训练一个模型来自动优化原始指令，使其能够更好地指导大语言模型完成各种任务。系统通过以下方式工作：

1. **指令重写**：模型接收原始指令，通过改写和添加特定要求生成优化后的指令
2. **奖励评估**：使用下游任务的实际表现（准确率、F1分数等）作为奖励信号
3. **强化学习训练**：采用 GRPO 算法优化重写模型，使其生成更有效的指令

### 支持的任务类型

- **数学问题**：GSM8K、MATH-500、AIME 2024
- **分类任务**：AG News、SST2、COPA、BoolQ、CB
- **问答任务**：NQ Open

### 支持的奖励函数

- `prewrite_accuracy_reward`：基于准确率的奖励（支持数学验证）
- `accuracy_math_reward`：数学问题准确率奖励
- `accuracy_classification_reward`：分类任务准确率奖励
- `f1_classification_reward`：分类任务 F1 分数奖励
- `accuracy_nq_open_instruction_prewrite_reward`：NQ Open 准确率奖励
- `f1_nq_open_instruction_prewrite_reward`：NQ Open F1 分数奖励
- `template_accuracy_math_reward`：模板化数学准确率奖励
- `accuracy_math_reward_split`：数学问题分割准确率奖励

## 安装

```shell
conda create -n PRewrite python=3.12
conda activate PRewrite
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

参考data/generate_data.ipynb

### 2. 配置训练参数

单阶段训练：参考 `examples/train.yaml` 配置文件

多阶段训练（重置模型参数和数据集）：参考 `examples/train_multi_stage.yaml` 配置文件

```yaml
# 模型配置
model_name_or_path: /path/to/base/model
output_dir: /path/to/output

# 数据集配置
dataset_name: /path/to/dataset
dataset_config: default
dataset_train_split: train
dataset_test_split: test

# 指令重写元提示
meta_instruction: "Rewrite the following instruction via rephrasing and/or adding specific requirements. Add instructions which would be helpful to solve the problem correctly. Output the new instruction only."

# 训练参数
learning_rate: 1e-6
gradient_checkpointing: true
dtype: bfloat16
max_prompt_length: 2048
max_completion_length: 2048

# 奖励函数配置
reward_funcs:
  - accuracy_classification_reward
  - f1_classification_reward
reward_weights:
  - 0.0
  - 1.0

# 生成模型配置（用于评估重写后的指令）
generate_model_name_or_path: /path/to/generate/model
generate_model_ip: 0.0.0.0
generate_model_port: 33337

# vLLM/SGLang 配置
use_vllm: true
vllm_mode: colocate
vllm_tensor_parallel_size: 8
vllm_gpu_memory_utilization: 0.25

# 生成参数
num_generations: 256
temperature: 1.0

# 训练参数
per_device_train_batch_size: 32
gradient_accumulation_steps: 1
loss_type: dapo
epsilon: 0.2
num_train_epochs: 1000
save_steps: 10
logging_steps: 1
```

### 3. 单阶段训练

使用 `examples/train.py` 进行单阶段训练：

```bash
export WANDB_PROJECT=PRewrite

accelerate launch \
    --config_file examples/deepspeed_zero3.yaml \
    examples/train.py --config examples/train.yaml \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
```

或使用提供的脚本：

```bash
bash scripts/train.sh
```

### 4. 多阶段训练

使用 `examples/train_multi_stage.py` 进行多阶段训练：

```bash
accelerate launch \
    --config_file examples/deepspeed_zero3.yaml \
    examples/train_multi_stage.py --config examples/train_multi_stage.yaml
```

或使用脚本：

```bash
bash scripts/train_multi_stage.sh
```

## 结果测试

参考`examples/model_test.py`，关键参数是`get_xx`函数中将`is_test`设置为`true`，就会对测试集的全集进行reward计算.

对于multi-stage训练，由于模型会重置，可以从log中取出rewrite的prompt，自己构建<prompt,dataset_name>对，再调用奖励函数进行计算，参考`examples/test_math_results.py`.

## 项目结构

```
PRewrite/
├── prewrite/              # 核心模块
│   ├── __init__.py
│   └── prewrite_rewards.py  # 奖励函数实现
├── examples/              # 训练脚本和配置
│   ├── train.py          # 单阶段训练脚本
│   ├── train_multi_stage.py  # 多阶段训练脚本
│   ├── train.yaml        # 训练配置文件
│   └── deepspeed_zero3.yaml  # DeepSpeed 配置
├── scripts/              # 训练脚本
│   ├── train.sh
│   └── train_multi_stage.sh
├── data/                 # 数据集目录
├── results/              # 训练结果
├── logs/                 # 训练日志
└── analysis/             # 分析和测试脚本
```

## 核心功能

### 奖励函数

奖励函数定义在 `prewrite/prewrite_rewards.py` 中，主要包括：

- **数学问题奖励**：使用 `math_verify` 进行数学表达式验证，或使用文本匹配
- **分类任务奖励**：基于准确率和 F1 分数
- **问答任务奖励**：基于 F1 分数和准确率

### 训练流程

1. **指令生成**：使用当前模型为每个样本生成重写后的指令
2. **指令评估**：使用生成模型（如 Qwen3-32B）在重写后的指令上完成任务
3. **奖励计算**：根据任务表现计算奖励
4. **策略优化**：使用 GRPO 算法更新模型参数

## 注意事项

1. **显存管理**：大模型训练需要足够的 GPU 显存，建议使用梯度检查点（`gradient_checkpointing: true`）和 DeepSpeed Zero3
2. **生成模型服务**：需要单独启动生成模型服务（使用 vLLM 或 SGLang），用于评估重写后的指令
3. **超参数调优**：根据具体任务调整 `num_generations`、`temperature`、`epsilon` 等参数
4. **奖励权重**：可以通过 `reward_weights` 调整不同奖励函数的权重
5. **多阶段训练**： 修改了`transformers/trainer.py`函数`_inner_training_loop`第2579行`epoch_dataloader = train_dataloader`为`epoch_dataloader = self.callback_handler.train_dataloader`，实现在训练过程中更新数据集。
