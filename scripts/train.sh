#! /bin/bash
base_options="--data-path /workspace/longtext-2k-clean.jsonl \
--tokenizer-path /workspace/tokenizer.model \
--output-path /root/autodl-tmp/output \
--ckpt-path /root/autodl-tmp/gemma-2b-it.ckpt
"

options="$base_options \
    --experiment-name train_pi_test \
    --epochs 2 \
    --batch-size-per-gpu 2 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.02 \
    --device cuda \
    --num-stages 8 \
    --max-len 16384 \
    --max-src-len 16000 \
    --seed 42 \
    --read-nums 1500 \
    --ds-config-path /workspace/gemma/gemma/ds_config/pineline.json \
    --variant 2b \
    --train_pi 2 \
    --lr 2e-5 \
    --warmup_min_lr 1e-6 \
    --warmup_max_lr 2e-5 \
    --use_lora \
    "

run_cmd="deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 16666 /workspace/gemma/gemma/train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x