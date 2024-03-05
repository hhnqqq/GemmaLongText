# @created by: haonan he
# @modified by: haonan he, zhijian jiang
import os
import time
import random

import math
import torch
import deepspeed
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

from model import GemmaForCausalLM, precompute_freqs_cis
from tokenizer import Tokenizer
from parser import base_parser, train_parser, ds_parser
from dataset import LongRopeDataset
from utils import  Timer, print_rank_0, read_config, ensure_directory_exists
from config import *
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
        # attention mask和input还需要处理一下, [batch_size, input_len, 1]
        input_ids, labels = inputs
        # 经过embedder计算, [batch_size, input_len, hidden_size]
        hidden_states = F.embedding(input_ids, self.weight)
        # gemma要使用hidden size对embedding输出进行正则化
        hidden_states = hidden_states * (args.hidden_size**0.5)
        # 获得attention mask, 这里还需要验证一下
        attention_mask = get_masks(input_ids.shape[1], device=hidden_states.device)
        # 获得rope频率
        freqs_cis = precompute_freqs_cis(args.head_dim,
                                         input_ids.shape[1],
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi).to(hidden_states.device)
        return hidden_states, freqs_cis, attention_mask, labels

class DecoderPipelineLayer(torch.nn.Module):
    # 还需要对k,v cache进行处理
    def __init__(self, model: GemmaForCausalLM, layer_idx):
        super().__init__()
        self.layer = model.model.layers[layer_idx]

    def forward(self, inputs):
        hidden_states, freqs_cis, attention_mask, labels = inputs
        # [batch_size, input_len, hidden_dim]
        hidden_states = self.layer(hidden_states, freqs_cis, attention_mask)
        return hidden_states, freqs_cis, attention_mask, labels
    
class FNormPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM):
        super().__init__()
        self.final_norm = model.model.norm
        self.emb_weight = model.embedder.weight.t()

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        # [batch_size, input_len, hidden_dim]
        hidden_states = self.final_norm(hidden_states)
        logits = torch.matmul(hidden_states, self.emb_weight.to(hidden_states.device).to(hidden_states.dtype))
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
        print(logits.shape, labels.shape)
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
    layers = [TiedLayerSpec("Embedding", EmbeddingPipelineLayer, model=model, args=args),
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
model = GemmaForCausalLM(model_config)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)
args.head_dim = model_config.head_dim
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers

model_pipe = PipelineModule(layers=get_model(model, args), num_stages=args.num_stages)
model_pipe.to(device).half()
tokenizer = Tokenizer(args.tokenizer_path)
train_dataset = LongRopeDataset(args.data_path, tokenizer, args.max_len, args.max_src_len, args.read_nums)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
ds_config['train_micro_batch_size_per_gpu'] = args.batch_size_per_gpu
ds_config['lr'] = args.lr

g = torch.Generator()
train_dataloader = DataLoader(train_dataset,
                            collate_fn=data_collator,
                            shuffle=True,
                            drop_last=True,
                            batch_size=args.batch_size_per_gpu,
                            generator=g)
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
print_rank_0("--->len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
print_rank_0("--->len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)
print_rank_0("--->args.batch_size_per_gpu = {}".format(args.batch_size_per_gpu), args.global_rank)
print_rank_0("--->args.num_update_steps_per_epoch = {}".format(num_update_steps_per_epoch), args.global_rank)
# start tranning

train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
engine, _, _, _ = deepspeed.initialize(model=model_pipe, config=ds_config, model_parameters=model_pipe.parameters())
all_loss = 0.0
with Timer() as timer:
    for step in range(args.epochs * num_update_steps_per_epoch):
        loss = engine.train_batch(data_iter=train_dataloader)
        print_rank_0("step = {}, loss = {}".format(step, loss.item()), args.global_rank)
        all_loss += loss.item()
        if args.local_rank == 0:
            if (step + 1) % args.show_loss_step == 0:
                now = time.time()
                avg_time = (now - timer.start) / args.show_loss_step
                avg_loss = all_loss / args.show_loss_step
                print(f"Step={step:>6}, loss={avg_loss:.4f}, {avg_time:.2f} it/s")
                start = now
                all_loss = 0.0

        if (step + 1) % args.save_interval == 0:
            print(f"Saving at step {step}")
            ensure_directory_exists(args.output_path)
            engine.save_checkpoint(args.output_path)
    ensure_directory_exists(args.output_path)
    engine.save_checkpoint(args.output_path)

print_rank_0(f"--->total time cosumed is {timer.time_cost}")
