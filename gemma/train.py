import os
import time
import random

import math
import torch
import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

from model import GemmaForCausalLM, Linear, LinearWithLoRA, precompute_freqs_cis
from tokenizer import Tokenizer
from parser import base_parser, train_parser, ds_parser
from dataset import LongRopeDataset
from utils import  Timer, print_rank_0, read_config, ensure_directory_exists
from utils.params_manager import refresh_config, print_trainable_module_names, enable_trainable_params
from config import *

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboard import SummaryWriter

# class and function define
class EmbeddingPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM, args):
        super().__init__()
        self.args = args
        self.embedder = model.embedder
        self.weight = self.embedder.weight
        # if args.quant:
        #     self.weight_scaler = self.word_embeddings.weight_scaler

    def forward(self, inputs):
        input_ids, labels = inputs
        # 经过embedder计算, [batch_size, input_len, hidden_size]
        hidden_states = F.embedding(input_ids, self.weight)
        # gemma要使用hidden size对embedding输出进行正则化
        hidden_states = hidden_states * (torch.tensor(args.hidden_size)**0.5)
        attention_mask = get_masks(input_ids.shape[1], device=hidden_states.device)
        # 获得rope频率
        freqs = precompute_freqs_cis(args.head_dim,
                                         input_ids.shape[1],
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi).to(hidden_states.device)
        freqs.requires_grad_(True)
        attention_mask.requires_grad_(True)
        return hidden_states, freqs, attention_mask, labels

class DecoderPipelineLayer(torch.nn.Module):
    # 还需要对k,v cache进行处理
    def __init__(self, model: GemmaForCausalLM, layer_idx):
        super().__init__()
        self.layer = model.model.layers[layer_idx]

    def forward(self, inputs):
        hidden_states, freqs, attention_mask, labels = inputs
        # [batch_size, input_len, hidden_dim]
        hidden_states = self.layer(hidden_states, freqs, attention_mask)
        return hidden_states, freqs, attention_mask, labels
    
class FNormPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM):
        super().__init__()
        self.final_norm = model.model.norm
        self.emb_weight = model.embedder.weight.t()

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        # [batch_size, input_len, hidden_dim]
        logits = self.final_norm(hidden_states)
        logits = torch.matmul(logits, self.emb_weight.to(hidden_states.device).to(hidden_states.dtype))
        return logits, labels
    
class SamplerPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM):
        super().__init__()
        self.embedder = model.embedder
        self.emb_weight = self.embedder.weight

    def forward(self, inputs):
        hidden_states, labels = inputs
        # [batch_size, input_len, vocab_size]
        logits = torch.matmul(hidden_states, self.emb_weight.t())
        return logits, labels

class LossPipelineLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        logits, labels = inputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_masks(seq_len, device):
    attention_mask = torch.full((1, 1, seq_len, seq_len),
                -2.3819763e38).to(torch.float)
    attention_mask = torch.triu(attention_mask, diagonal=1).to(device)
    return attention_mask

def get_model(model, args):
    layers = [LayerSpec(EmbeddingPipelineLayer, model=model, args=args),
              *[LayerSpec(DecoderPipelineLayer, model=model, layer_idx=idx) for idx in
                range(args.num_layers)],
              LayerSpec(FNormPipelineLayer, model=model),
              LayerSpec(LossPipelineLayer)]
    return layers

def data_collator(examples):
    input_ids_list, labels_list = [], []
    for instance in examples:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["labels"])
    return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))

def switch_to_lora(model, replace_names, rank=4, lora_scaler=32):
    for name, module in model.named_modules():
        for replace_name in replace_names:
            if isinstance(module, Linear) and replace_name in name:
                # 创建LinearWithLoRA实例
                lora_layer = LinearWithLoRA(rank, lora_scaler, module.in_features, module.out_features, module.quant)
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


# load args
args = ds_parser(train_parser(base_parser())).parse_args()

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
args.global_rank = torch.distributed.get_rank()
set_random_seed(args.seed)

# load model and dataset
model_config = get_model_config(args.variant)
print_rank_0('--->loading the model')
model = GemmaForCausalLM(model_config)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)
if args.use_lora:
    switch_to_lora(model, ['qkv_proj'], rank=8)
enable_trainable_params(model, ['weight_a','weight_b'])
print_trainable_module_names(model)
args.head_dim = model_config.head_dim
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers

model_pipe = PipelineModule(layers=get_model(model, args), num_stages=args.num_stages)
model_pipe.to(device).half()
tokenizer = Tokenizer(args.tokenizer_path)
train_dataset = LongRopeDataset(args.data_path, tokenizer, args.max_len, args.max_src_len, args.read_nums)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

g = torch.Generator()
train_dataloader = DataLoader(train_dataset,
                            collate_fn=data_collator,
                            shuffle=True,
                            drop_last=True,
                            batch_size=args.batch_size_per_gpu,
                            generator=g)
assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    num_update_steps = args.epochs * (math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
else:
    num_update_steps = args.train_iters/args.gradient_accumulation_steps
num_warmup_steps = int(num_update_steps * args.warmup)
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
print_rank_0("--->TRAIN DATALOADER LENGTH: len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
print_rank_0("--->TRAIN DATASET LENGTH: = {}".format(len(train_dataset)), args.global_rank)
print_rank_0("--->TRAIN BATCH SIZE PER GPU: args.batch_size_per_gpu = {}".format(args.batch_size_per_gpu), args.global_rank)
print_rank_0("--->NUMBER OF UPDATE STEPS: args.num_update_steps = {}".format(num_update_steps), args.global_rank)
print_rank_0("--->NUMBER OF WARMUP STEPS: args.num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
# start tranning

train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=ds_config, model_parameters=[p for p in model_pipe.parameters() if p.requires_grad])
all_loss = 0.0
print_rank_0('--->loaded the model, start training')
with Timer() as timer:
    for step in range(num_update_steps):
        loss = engine.train_batch(data_iter=train_dataloader)
        print_rank_0("--->step={}, loss={}".format(step, loss.item()), args.global_rank)
        all_loss += loss.item()
        if args.local_rank == 0:
            if (step + 1) % args.show_loss_step == 0:
                now = time.time()
                avg_time = (now - timer.start) / args.show_loss_step
                avg_loss = all_loss / args.show_loss_step
                print(f"--->step={step:>6}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
                start = now
                all_loss = 0.0

        if (step + 1) % args.save_interval == 0:
            print(f"Saving at step {step}")
            ensure_directory_exists(args.output_path)
            engine.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-{step}')
    ensure_directory_exists(args.output_path)
    engine.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-final')

print_rank_0(f"--->total time cosumed is {timer.time_cost}")