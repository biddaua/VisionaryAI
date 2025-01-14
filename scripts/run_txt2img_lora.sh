
#!/bin/bash


export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=7; \
python txt2img.py \
    --prompt "一个带着红色蝴蝶结的小女孩" \
    --config configs/v1-inference-chinese-lora.yaml \
    --output_path ./output/ \
    --enable_lora True \
    --lora_ckpt_filepath ./output/txt2img_lora/ckpt/rank_0/wkhh_txt2img_lora-12_1224.ckpt \
    --seed 42 \
    --n_iter 4 \
    --n_samples 4 \
    --W 512 \
    --H 512 \
    --ddim_steps 30 > test_lora.log 2>&1 &
