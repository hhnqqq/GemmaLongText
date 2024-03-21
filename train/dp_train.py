import math
import torch
import deepspeed
import numpy as np
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from gemma.config import *
from gemma.tokenizer import Tokenizer
from gemma.parser import base_parser, train_parser, ds_parser
from gemma.dataset import LongRopeDataset
from gemma.model import GemmaForCausalLM, Linear, LinearWithLoRA, precompute_freqs_cis
from gemma.utils.optimizer import get_optimizer
from gemma.utils import Timer, DataCollator, print_rank_0, read_config, ensure_directory_exists, to_device, get_masks, set_random_seed
from gemma.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)

class TrainModel(torch.nn.Module):
    def __init__(self, model:GemmaForCausalLM, args):
        super().__init__()
        self.args = args
        self.model = model.model
        self.embedder = model.embedder
        self.emb_weight = model.embedder.weight
    
    def forward(self, input_ids, labels):
        hidden_states = F.embedding(input_ids, self.emb_weight)
        hidden_states = hidden_states * (torch.tensor(args.hidden_size)**0.5)
        freqs_cis = precompute_freqs_cis(args.head_dim,
                                         input_ids.shape[1],
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi).to(hidden_states.device)
        attention_mask = get_masks(input_ids.shape[1], device=hidden_states.device)
        logits = self.model(hidden_states=hidden_states, freqs_cis=freqs_cis, mask=attention_mask)
        logits = torch.matmul(logits, self.emb_weight.t().to(hidden_states.device).to(hidden_states.dtype))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss

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
model_config = get_model_config(args.variant)
args.head_dim = model_config.head_dim
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
args.gpu_count = torch.cuda.device_count()
args.global_rank = torch.distributed.get_rank()
set_random_seed(args.seed)

# load model and dataset
print_rank_0('--->loading the model', args.global_rank)
model = GemmaForCausalLM(model_config)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)
model = TrainModel(model, args)
if args.use_lora or args.use_lora_plus:
    if args.replace_modules is None:
        args.replace_modules = ['qkv_proj']
    switch_to_lora(model, args.replace_modules, rank=4)
    enable_trainable_params(model, ['weight_a','weight_b'])
elif args.disable_list is not None:
    disable_untrainable_params(model, args.disable_list)
elif args.enable_list is not None:
    enable_trainable_params(model, args.enable_list)
print_trainable_module_names(model)

if args.fp16:
    model.to(device).half()
elif args.bf16:
    model.to(device).bfloat16()

tokenizer = Tokenizer(args.tokenizer_path)
train_dataset = LongRopeDataset(args.data_path, tokenizer, args.max_len, args.max_src_len, args.read_nums)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

if args.local_rank == -1:
    train_sampler = RandomSampler(train_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
data_collator = DataCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                batch_size=args.batch_size_per_gpu)

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    args.num_update_steps = args.epochs * (math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps)))
else:
    args.num_update_steps = args.train_iters/args.gradient_accumulation_steps
args.num_warmup_steps = int(args.num_update_steps * args.warmup) + 1
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
print_rank_0("--->TRAIN DATALOADER LENGTH: len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
print_rank_0("--->TRAIN DATASET LENGTH: = {}".format(len(train_dataset)), args.global_rank)
print_rank_0("--->TRAIN BATCH SIZE PER GPU: args.batch_size_per_gpu = {}".format(args.batch_size_per_gpu), args.global_rank)
print_rank_0("--->NUMBER OF UPDATE STEPS: args.num_update_steps = {}".format(args.num_update_steps), args.global_rank)
print_rank_0("--->NUMBER OF WARMUP STEPS: args.num_warmup_steps = {}".format(args.num_warmup_steps), args.global_rank)

optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model)
model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, 
                                                         args=args, 
                                                         optimizer=optimizer,
                                                         lr_scheduler=lr_scheduler,
                                                         config=ds_config,
                                                         dist_init_required=True)
model.train()
tr_loss, min_loss = 0.0, 0.0
global_step = 0
# train
ensure_directory_exists(args.output_path)
with Timer() as timer:
    for epoch in range(args.epochs):
        print_rank_0(f"--->Beginning of Epoch {epoch + 1}/{args.epochs}, Total Micro Batches {train_dataloader}" ,args.global_rank)
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            timer.average_time(entry='start')
            batch = to_device(batch, device)
            loss = model(**batch)
            tr_loss += loss.item()
            model.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # write loss
                if global_step % args.show_loss_step == 0:
                    timer.average_time(entry='end')
                    avg_time = (timer.loop_time) / args.show_loss_step
                    avg_loss = tr_loss / args.show_loss_step
                    print_rank_0(f"--->Epoch: {epoch}, step: {step + 1}, global_step:{global_step}, avg_loss: {avg_loss:.2f}, avg_time: {avg_time}s"
                                 ,args.global_rank)
                    tr_loss = 0.0
                # save model
                if args.save_interval is not None and global_step % args.save_interval == 0:
                    # 若zero3训练，模型参数需要合并保存
                    if ds_config["zero_optimization"]["stage"] == 3:
                        state_dict = model._zero3_consolidated_16bit_state_dict()
                        if args.global_rank <= 0:
                            model.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-{step}')
                    else:
                        if args.global_rank <= 0:
                            model.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-{step}')
                    model.train()

    if ds_config["zero_optimization"]["stage"] == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
        if args.global_rank <= 0:
            model.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-final')
    else:
        if args.global_rank <= 0:
            model.save_checkpoint(args.output_path, tag=f'{args.experiment_name}-final')

print_rank_0(f"--->total time cosumed is {timer.time_cost}", args.global_rank)
