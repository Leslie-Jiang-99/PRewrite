export WANDB_PROJECT=PRewrite

accelerate launch \
    --config_file examples/deepspeed_zero3_4gpu.yaml \
    examples/train.py --config examples/train_4gpu.yaml \
    2>&1 | tee -a /root/workspace/PRewrite/logs/4gpu_math_500_only_$(date +%Y%m%d_%H%M%S).log