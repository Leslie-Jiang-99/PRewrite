export WANDB_PROJECT=PRewrite

accelerate launch \
    --config_file examples/deepspeed_zero3_4gpu.yaml \
    examples/train.py --config examples/train_4gpu.yaml \
    2>&1 | tee -a /root/workspace/PRewrite/logs/ag_news_only_64_$(date +%Y%m%d_%H%M%S).log