#!/bin/bash


export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
device_id=2

output_path=output/
task_name=txt2img
data_path=dataset/
pretrained_model_path=models/
train_config_file=configs/train_config.json

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \
nohup python -u run_train.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    > $output_path/$task_name/log_train 2>&1 &