export WANDB_PROJECT=PRewrite

accelerate launch \
    --config_file examples/deepspeed_zero3.yaml \
    examples/train_multi_stage.py --config examples/train_multi_stage.yaml \
    2>&1 | tee -a /root/workspace/PRewrite/logs/extended_dataset_only_classification_multi_stage_80_$(date +%Y%m%d_%H%M%S).log