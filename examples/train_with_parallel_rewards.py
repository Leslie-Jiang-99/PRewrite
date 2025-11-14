"""
使用并行reward计算的训练脚本示例

修改自原始的train.py，添加了并行reward计算功能
"""

# 在导入GRPOTrainer之前，先patch它
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ============ 选择你的并行化方案 ============
# 方案1：ThreadPoolExecutor（推荐，简单稳定）
from parallel_reward_patch import patch_grpo_trainer
patch_grpo_trainer()

# 方案2：asyncio（更高效，但需要配置nest_asyncio）
# from parallel_reward_async import patch_grpo_trainer_async
# patch_grpo_trainer_async()
# ==========================================

# 然后继续正常导入
from trl import GRPOTrainer, GRPOConfig
# ... 其他导入


# ============ 性能对比测试 ============
def benchmark_reward_computation():
    """
    测试串行 vs 并行的性能差异
    """
    import time
    import torch
    from trl import GRPOTrainer
    
    # 模拟3个需要2秒的reward functions
    def slow_reward_func_1(prompts, completions, **kwargs):
        time.sleep(2.0)  # 模拟网络IO
        return [1.0] * len(prompts)
    
    def slow_reward_func_2(prompts, completions, **kwargs):
        time.sleep(2.0)
        return [0.5] * len(prompts)
    
    def slow_reward_func_3(prompts, completions, **kwargs):
        time.sleep(2.0)
        return [0.8] * len(prompts)
    
    reward_funcs = [slow_reward_func_1, slow_reward_func_2, slow_reward_func_3]
    
    # 模拟数据
    prompts = ["test prompt"] * 10
    completions = ["test completion"] * 10
    
    print("=" * 60)
    print("🔬 Benchmarking Reward Computation")
    print("=" * 60)
    
    # 测试串行版本
    print("\n📊 Testing SERIAL execution...")
    start = time.time()
    for func in reward_funcs:
        func(prompts, completions)
    serial_time = time.time() - start
    print(f"   ⏱️  Serial time: {serial_time:.2f} seconds")
    
    # 测试并行版本
    print("\n📊 Testing PARALLEL execution...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(func, prompts, completions) for func in reward_funcs]
        results = [f.result() for f in as_completed(futures)]
    parallel_time = time.time() - start
    print(f"   ⏱️  Parallel time: {parallel_time:.2f} seconds")
    
    print(f"\n✨ Speedup: {serial_time / parallel_time:.2f}x faster!")
    print(f"   Time saved: {serial_time - parallel_time:.2f} seconds per batch")
    print("=" * 60)


if __name__ == "__main__":
    # 运行性能测试
    # benchmark_reward_computation()
    
    # 正常训练
    # main(script_args, training_args, model_args, dataset_args)
    pass

