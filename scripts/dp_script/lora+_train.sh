#! /bin/bash
base_options="--data-path /workspace/longtext-2k-clean.jsonl \
--tokenizer-path /workspace/tokenizer.model \
--output-path /workspace/gemma/output \
--ckpt-path /workspace/gemma-2b-it.ckpt
"

options="$base_options \
    --experiment-name train_pi_test \
    --show-loss-step 1 \
    --epochs 3 \
    --batch-size-per-gpu 1 \
    --fp16 \
    --gradient-accumulation-steps 1 \
    --warmup 0.02 \
    --device cuda \
    --max-len 16384 \
    --max-src-len 16000 \
    --seed 42 \
    --read-nums 100 \
    --ds-config-path /workspace/gemma_long_rope/gemma/ds_config/stage2.json \
    --variant 2b \
    --train-pi 2 \
    --lr 1e-5 \
    --warmup-min-lr 1e-6 \
    --warmup-max-lr 2e-5 \
    --use-lora-plus \
    --diy-optimizer \
    --atten-type flash_atten \
    --activation-checkpoint \
    "
    
run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /workspace/gemma_long_rope/train/dp_train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x