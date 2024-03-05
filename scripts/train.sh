#! /bin/bash
base_options="--data-path /workspace/gemma/data/LongQLoRA-SFT-Data-39k.jsonl \
--tokenizer-path /workspace/gemma/data/tokenizer.model \
--output-path /workspace/gemma/output/ \
"

options="$base_options \
    --experiment-name train_pi_test \
    --epochs 1 \
    --batch-size-per-gpu 8 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.02 \
    --device cuda \
    --num-stages 1 \
    --max-len 768 \
    --max-src-len 256 \
    --seed 42 \
    --read-nums 1000 \
    --ds-config-path /workspace/gemma/gemma/ds_config/pineline.json \
    --variant test \
    --train_pi \
    "

run_cmd="CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 16666 /workspace/gemma/gemma/train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x