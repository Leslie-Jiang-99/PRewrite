export WANDB_PROJECT=PRewrite

accelerate launch \
    --config_file examples/deepspeed_zero3.yaml \
    examples/train.py --config examples/train.yaml \
    2>&1 | tee -a /root/workspace/PRewrite/logs/ag_news_only_256_1_$(date +%Y%m%d_%H%M%S).log