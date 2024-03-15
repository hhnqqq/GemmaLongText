# 使用deepspeed流水线并行构建一个支持长文本的gemma-2b

### 环境配置
- 下载好预训练checkpoint: [gemma.ckpt](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch)
- 准备一台有足够卡的服务器，笔者的配置是一台租用的8000服务器，cuda版本最好够新
- 安装本仓库要求的依赖：pip install -r requirements.txt
- 安装gemma的配置文件：python setup.py install
- 准备好长文本数据集

### 使用方法
编辑gemma/scripts/train.sh
```bash
#! /bin/bash
base_options="--data-path /workspace/longtext-2k-clean.jsonl \
--tokenizer-path /workspace/tokenizer.model \
--output-path /workspace/gemma/output \
--ckpt-path /workspace/gemma-2b-it.ckpt
"

# disable_list=("embedder","mlp")

options="$base_options \
    --experiment-name train_pi_test \
    --show-loss-step 1 \
    --epochs 3 \
    --batch-size-per-gpu 1 \
    --fp16 \
    --gradient-accumulation-steps 2 \
    --warmup 0.02 \
    --device cuda \
    --num-stages 7 \
    --max-len 15000 \
    --max-src-len 14000 \
    --seed 42 \
    --read-nums 100 \
    --ds-config-path /workspace/gemma/gemma/ds_config/pineline.json \
    --variant 2b \
    --train-pi 2 \
    --lr 2e-5 \
    --warmup-min-lr 1e-6 \
    --warmup-max-lr 2e-5 \
    --use-lora \
    --activation-checkpoint \
    "

# for item in "${disable_list[@]}"; do
#     options+=" \"$item\""
# done

run_cmd="deepspeed --include localhost:0,1,2,3,4,5,6 --master_port 16666 /workspace/gemma/gemma/train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}
```
