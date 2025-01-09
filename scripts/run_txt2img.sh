
#!/bin/bash


export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=0; \
python txt2img.py \
    --prompt "来自深渊 风景 绘画 写实风格" \
    --config configs/v1-inference-chinese.yaml \
    --output_path ./output/ \
    --seed 42 \
    --dpm_solver \
    --n_iter 4 \
    --n_samples 4 \
    --W 512 \
    --H 512 \
    --ddim_steps 15
