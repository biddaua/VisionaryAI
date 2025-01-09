#!/bin/bash


export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=0; \
python inpaint.py \
    --prompt "一只红色的狐狸坐在长椅上" \
    --img demo/inpaint/overture-creations-5sI6fQgYIuo.png \
    --mask demo/inpaint/overture-creations-5sI6fQgYIuo_mask.png \
    --config configs/wukong-huahua_inpaint_inference.yaml \
    --ckpt_name wukong-huahua-inpaint-ms.ckpt 