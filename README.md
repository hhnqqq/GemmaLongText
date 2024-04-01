# Building a Long Text Supported Model with Deepspeed for GEMMA

[中文](https://github.com/hhnqqq/GemmaLongText/blob/main/README_ZH.md)

### Environment Setup
- Download the pre-trained checkpoint: [gemma.ckpt](https://www.kaggle.com/models/google/gemma/frameworks/pyTorch)
- Prepare a server with sufficient GPU power, my setup is an 8000 server, with preferably a new CUDA version
- Install the configuration files of this repository, dependencies will be automatically configured: python setup.py install
- Prepare a dataset with long text

### 使用方法
- Choose the fine-tuning method to use, lora/lora+/galore/full
- Choose the parallel training method to use: pp/dp
- Customize the corresponding script based on your local conditions, for example, when using lora and pp, edit /scripts/pp_scrtpt/lora_train.sh
- Start by editing the settings in base options
    - Replace data-path with your dataset address, the dataset should be in jsonl format, containing input and output keys
    - Replace output-path with the save location of your model checkpoint
    - Replace ckpt-path with the address of your pre-trained model
- Edit the settings in options based on your training requirements

```bash
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
    --gradient-accumulation-steps 2 \
    --warmup 0.02 \
    --device cuda \
    --num-pp-stages 4 \
    --max-len 16384 \
    --max-src-len 16000 \
    --seed 42 \
    --read-nums 100 \
    --ds-config-path /workspace/gemma_long_rope/gemma/ds_config/pipeline.json \
    --variant 2b \
    --train-pi 2 \
    --lr 1e-5 \
    --warmup-min-lr 1e-6 \
    --warmup-max-lr 2e-5 \
    --use-lora-plus \
    --activation-checkpoint \
    --diy-optimizer \
    --atten-type flash_atten \
    "

run_cmd="deepspeed --include localhost:0,1,2,3 --master_port 16666 /workspace/gemma_long_rope/gemma/train.py ${options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
```
### NEWS
- Added support for activation checkpoint
- Added support for custom optimizer
- Added support for lr scheduler from sat library
- Added support for lora+ (applying different learning rates to matrix A and B of lora can lead to better performance and faster convergence)
- 2024-03-20 Added support for galore(issues with gradient dimensions, to be modified)
- 2024-03-21 Added support for torch's flash-attention implementation
- 2024-03-21 Added support for pure dp training
- 2024-03-23 Added support for lora-fa
- <b>2024-04-01 Added support for deepspeed-ulysses, now able to train longer models!!</b>

### TODO
- Support more lora versions like dora
- Support more length extrapolation methods
- Support more memory-efficient methods
- And more advanced technologies
- Enhance the scheduler code
- Add support for ring-attention

### Acknowledgements

Thanks to the open-source code and model weights from the following repositories:
- [sat](https://github.com/THUDM/SwissArmyTransformer)
- [gemma](https://github.com/google/gemma_pytorch)
- [loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus)
- [chatglm-finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
- And other repositories I referenced and borrowed from

### 联系

Feel free to contact me via:
- e-mail：hnhe@mail.ustc.edu.cn


