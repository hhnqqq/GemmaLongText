import math
import torch
import deepspeed
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import gemma.utils.parallel_states as parallel_states

from gemma.config import *
from gemma.tokenizer import Tokenizer
from gemma.parser import base_parser, train_parser, ds_parser
from gemma.dataset import LongRopeDataset
from gemma.model import GemmaForCausalLM, precompute_freqs_cis
from gemma.lora import LinearWithLoRA, switch_to_lora
from gemma.utils.optimizer import get_optimizer
from gemma.utils import Timer, DataCollator, print_rank_0, read_config, ensure_directory_exists, to_device, get_masks, set_random_seed
from gemma.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)

class TrainModel(torch.nn.Module):
    """
    Trainer class for Gemma, responsible for handling input and output during training.
    """
    def __init__(self, model:GemmaForCausalLM, args, pad_id):
        """
        Initializes basic attributes for the trainer class and precomputes fixed values.

        param model: Gemma model with pretrained weight.
        param args: Arguments from argument parser.
        param pad_id: Pad index of the tokenizer, used to set ignore index for loss function.
        """
        super().__init__()
        self.args = args
        self.model = model.model
        self.embedder = model.embedder
        self.emb_weight = model.embedder.weight
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
        self.attention_mask = get_masks(args.max_len)
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                         args.max_len,
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi)
    
    def forward(self, input_ids, labels):
        seq_parallel_world_size = parallel_states.get_sequence_parallel_world_size()
        seq_parallel_world_rank = parallel_states.get_sequence_parallel_rank()
        if args.atten_type is not None and 'ulysses' in args.atten_type:
            assert args.max_len % seq_parallel_world_size == 0, 'Max input length is not divisble by sequence parallel stages.'
            assert args.head_nums % seq_parallel_world_size == 0, 'Attention head num is not divisble by sequence parallel stages.'
            # Split the input ids and lables and freqs cis for deepspeed-ulysses.
            seq_len_per_group = args.max_len // seq_parallel_world_size
            local_seq_start = seq_parallel_world_rank * seq_len_per_group
            local_seq_end = (seq_parallel_world_rank +1) * seq_len_per_group
            input_ids = input_ids[:, local_seq_start:local_seq_end]
            labels = labels[:, local_seq_start:local_seq_end]
            freqs_cis = self.freqs_cis[local_seq_start:local_seq_end,:].to(input_ids.device)
        else:
            freqs_cis = self.freqs_cis.to(input_ids.device)
        hidden_states = F.embedding(input_ids, self.emb_weight)
        hidden_states = hidden_states * (torch.tensor(args.hidden_size)**0.5)
        attention_mask = self.attention_mask.to(hidden_states.device).to(hidden_states.dtype)
        # Using activation checkpoint to reduce memory consumption or not.
        freqs_cis.requires_grad_(True)
        if self.args.activation_checkpoint:
            logits = checkpoint(self.model, hidden_states, freqs_cis, attention_mask, self.args.atten_type)
        else:
            logits = self.model(hidden_states=hidden_states, freqs_cis=freqs_cis, mask=attention_mask, atten_type=self.args.atten_type)
        logits = torch.matmul(logits, self.emb_weight.t().to(hidden_states.device).to(hidden_states.dtype))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss

# Load arguments
args = ds_parser(train_parser(base_parser())).parse_args()
model_config = get_model_config(args.variant)
args.head_dim = model_config.head_dim
args.head_num = model_config.num_attention_heads
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    args.world_size = dist.get_world_size()
    args.global_rank = dist.get_rank()
if args.num_sp_stages is not None:
    assert args.atten_type == 'ulysses_atten', 'when using sequence parallism, the attention type must be `ulysses_atten`'
    parallel_states.initialize_model_parallel(sequence_model_parallel_size=args.num_sp_stages)
else:
    parallel_states.initialize_model_parallel()
set_random_seed(args.seed)

# load model and dataset
print_rank_0('--->Loading the model.', args.global_rank)
tokenizer = Tokenizer(args.tokenizer_path)
model = GemmaForCausalLM(model_config)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)
model = TrainModel(model, args, tokenizer.pad_id)

if args.use_lora or args.use_lora_plus:
    switch_to_lora(model, args.replace_modules, rank=args.lora_rank, use_dora=args.use_dora)
    if args.lora_fa:
        enable_trainable_params(model, ['weight_b'])
    else:
        enable_trainable_params(model, ['weight_a', 'weight_b'])
elif args.disable_list or args.enable_list:
    param_list = args.disable_list if args.disable_list is not None else args.enable_list
    disable_untrainable_params(model, param_list) if args.disable_list else enable_trainable_params(model, param_list)

if args.fp16:
    model.to(device).half()
elif args.bf16:
    model.to(device).bfloat16()

train_dataset = LongRopeDataset(args.data_path, tokenizer, args.max_len, args.max_src_len, args.mode, args.read_nums)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

# TODO: 这里的处理还有明显的问题，需要修改
if args.local_rank == -1 or args.num_sp_stages is not None:
    train_sampler = RandomSampler(train_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
data_collator = DataCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                batch_size=args.batch_size_per_gpu)

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    # TODO: 修正sp与dp时的不同
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
                                                         optimizer=optimizer,
                                                         lr_scheduler=lr_scheduler,
                                                         config=ds_config,
                                                         model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                         mpu=parallel_states)
print_trainable_module_names(model)
tr_loss, min_loss = 0.0, 0.0
global_step = 0
# train
ensure_directory_exists(args.output_path)
with Timer() as timer:
    for epoch in range(args.epochs):
        print_rank_0(f"--->Beginning of Epoch {epoch + 1}/{args.epochs}, Total Micro Batches {len(train_dataloader)}" ,args.global_rank)
        
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