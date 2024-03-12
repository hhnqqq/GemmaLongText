# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import contextlib
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch

from tqdm import tqdm
from gemma import config
from gemma import model as gemma_model


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def switch_to_lora(model, replace_names, rank=4, lora_scaler=32):
    total = sum([1 for _, _ in model.named_modules()])
    for name, module in tqdm(model.named_modules(), total=total):
        for replace_name in replace_names:
            if isinstance(module, gemma_model.Linear) and replace_name in name:
                # 创建LinearWithLoRA实例
                lora_layer = gemma_model.LinearWithLoRA(rank, lora_scaler, module.in_features, module.out_features, module.quant)
                # 复制原始参数
                lora_layer.weight.data = module.weight.data
                if module.quant:
                    lora_layer.weight_scaler = module.weight_scaler
                # 用新层替换旧层
                parent = get_parent_model(model, module)
                setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], lora_layer)

def get_parent_model(parent_model, module):
    for _, sub_module in parent_model._modules.items():
        if sub_module is module:
            return parent_model
    for _, sub_module in parent_model._modules.items():
        parent = get_parent_model(sub_module, module)
        if parent:
            return parent
    return None

def main(args):
    # Construct the model config.
    model_config = config.get_model_config(args.variant)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(args.ckpt)
        switch_to_lora(model, ['gate_proj'])
        model = model.to(device).eval()
    print("Model loading done")
    print('======================================')
    print(f'The device of the model is {device}')
    print(f'The dtype of the model is {model_config.dtype}')
    print('======================================')
    # Print the prompts and results.

    if args.prompt is not None:
        result = model.generate(args.prompt, device, output_len=4096)
        print('======================================')
        print(f'PROMPT: {args.prompt}')
        print(f'RESULT: {result}')
        print('======================================')

    if args.run_loop:
        while True:
            prompt = str(input('===>Please enter your prompt, or enter quit() to quit: '))
            if prompt == 'quit()':
                break
            else:
                result = model.generate(prompt, device, output_len=4096)
                print('======================================')
                print(f'PROMPT: {prompt}')
                print(f'RESULT: {result}')
                print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='/workspace/gemma/data/gemma-2b.ckpt')
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--run_loop", action='store_true')
    parser.add_argument("--output_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    main(args)
