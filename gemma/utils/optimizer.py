import torch.optim as optim
from utils import print_rank_0
from utils.scheduler import AnnealingLR

def get_optimizer(ds_config, args, model):
    # TODO: 增加对DoRA, pLoRA等微调方法的支持
    if args.diy_optimizer or args.use_lora_plus:
        if args.optim_type is not None:
            optim_type = args.optim_type
        elif 'optimizer' in ds_config:
            optim_type = ds_config['optimizer'].get('type', 'adamw')

        if args.use_lora_plus:
            weight_b_group = [p for n, p in model.named_parameters() if p.requires_grad and 'weight_b' in n]
            base_group = [p for n, p in model.named_parameters() if p.requires_grad and 'weight_b' not in n]
            params = [{'params': weight_b_group, 'lr': args.lr * args.lora_plus_scaler},
                      {'params': base_group, 'lr': args.lr}]
            print_rank_0(F'--->lora+ is enabled and the lr of weight b is set to {args.lr * args.lora_plus_scaler}', args.global_rank)
        # elif args.use_dora:
        #     pass
        # elif args.use_plora:
        #     pass
        else:
            params = [p for p in model.parameters() if p.requires_grad]

        optimizer_class = {
            'adamw': optim.AdamW,
            'adam': optim.Adam
        }.get(optim_type.lower())
        
        if optimizer_class is None:
            raise NotImplementedError('only support adamw and adam optimizer now')
        
        optimizer = optimizer_class(params=params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps,
                                    betas=tuple(args.betas))
        
        lr_scheduler = get_learning_rate_scheduler(optimizer, 0, args)
        del ds_config['optimizer']
        print_rank_0(F'--->deepspeed optimizer setting have been overwritten', args.global_rank)
        return optimizer, lr_scheduler
    
    return None

def get_learning_rate_scheduler(optimizer, iteration, args,
                                auto_warmup_steps=100, auto_warmup_rate=0.05):
    init_step = max(iteration - auto_warmup_steps, 0)
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=args.num_warmup_steps,
                               num_iters=args.num_update_steps,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio,
                               auto_warmup_steps=auto_warmup_steps,
                               auto_warmup_rate=auto_warmup_rate
                               )

    return lr_scheduler