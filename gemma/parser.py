import argparse
import deepspeed

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',type=str, required=True)
    parser.add_argument('--ckpt-path',type=str, default=None)
    parser.add_argument('--tokenizer-path',type=str, required=True)
    parser.add_argument('--output-path',type=str, required=True)
    return parser

def train_parser(parser):
    group = parser.add_argument_group('train', 'training configurations')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--experiment-name', type=str, default="MyModel",
                       help="The experiment name for summary and checkpoint."
                       "Will load the previous name if mode==pretrain and with --load ")
    group.add_argument('--train-iters', type=int, default=None,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--epochs', type=int, default=None,
                       help='number of train epochs')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bf16 mode')
    group.add_argument('--variant', type=str, default='2b',choices=['test', '2b', '7b'],
                       help='the variant of the model.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--device', type=str, default='cpu',
                       help='the device to load the model')
    
    # --------------------- optimizer -----------------------
    group.add_argument('--diy-optimizer', action='store_true',
                       help='weather to diy the optimizer')
    group.add_argument('--batch-size-per-gpu', type=int, default=4,
                       help='batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--eps', type=float, default=1e-8,
                       help='initial eps for the optimizer')
    group.add_argument('--betas', nargs='+', type=float, default=[0.9,0.95],
                       help='initial eps for the optimizer')
    group.add_argument('--warmup-min-lr', type=float, default=1.0e-5,
                       help='minimum learning rate of warmup')
    group.add_argument('--warmup-max-lr', type=float, default=2.0e-4,
                       help='maxium learning rate of warmup')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='run optimizer after every gradient-accumulation-steps backwards.')
    group.add_argument('--auto-warmup-steps', type=int, default=10,
                       help='the fix warmup steps for training')
    group.add_argument('--auto-warmup-rate', type=float, default=0.05,
                       help='the warmup rate for fix warmup steps')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=5e-4,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--lr-decay-style', type=str, default='cosine',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--optim-type', type=str, default=None,
                       help='type of the optimizer')

    # ---------------------------- dataset ------------------------------
    group.add_argument('--read-nums', type=int, default=None,
                       help='the number of data to read')
    group.add_argument('--max-len', type=int, default=None,
                       help='max len of tokens')
    group.add_argument('--max-src-len', type=int, default=None,
                       help='max len of input tokens')
    
    # --------------------------- parameters ----------------------------
    group.add_argument('--enable-list', nargs='+', type=str, default=None, 
                       help='List of enable params')
    group.add_argument('--disable-list', nargs='+', type=str, default=None, 
                       help='List of disable params')
    group.add_argument('--activation-checkpoint', action='store_true', 
                       help='Train model with activation checkpoint')

    # --------------------------- lora ----------------------------------
    group.add_argument('--use-lora', action='store_true',
                       help='weather to use lora')
    group.add_argument('--use-lora-plus', action='store_true',
                       help='weather to use lora+')
    group.add_argument('--lora-rank', type=int, default=8,
                       help='the rank of lora')
    group.add_argument('--lora-plus-scaler', type=int, default=16,
                       help='the scaler of lora weight b')
    group.add_argument('--replace-modules', nargs='+', type=str, default=None,
                       help='List of modules to be replaced by lora')
    
    # --------------------------- galore ----------------------------------
    group.add_argument('--use-galore', action='store_true',
                    help='weather to use galore')
    group.add_argument('--galore-rank', type=int, default=8,
                       help='the rank of galore')
    group.add_argument('--galore-scaler', type=float, default=0.25,
                       help='the scaler of galore')
    group.add_argument('--galore-per-layer', action='store_true')

    # -------------------------- others ----------------------------
    group.add_argument('--seed', type=int, default=None,
                       help='random seed')
    group.add_argument('--show-loss-step', type=int, default=1)
    group.add_argument('--rope-theta', default=10000.0,
                       help='rope theta')
    group.add_argument('--train-pi', type=int, default=None,
                       help='In the case of a non-existent interpolation multiple, the rope will remain in its original state.')
    group.add_argument('--flash-atten', action='store_true',
                       help='weather to flash attention')

    return parser

def ds_parser(parser):
    group = parser.add_argument_group('ds', 'ds configurations')
    group.add_argument("--ds-config-path",type=str,
                      help="path of ds configuration file",)
    group.add_argument("--local_rank",type=int,default=-1,
                      help="local rank passed from distributed launcher",)
    group.add_argument("--global-rank", default=-1, type=int, 
                      help="global rank")
    group.add_argument("--with-aml-log", default=True, 
                      help="Use Azure ML metric logging")
    group.add_argument("--offload-optimizer", action='store_true')
    group.add_argument("--offload-param", action='store_true')
    group.add_argument("--csv-monitor", action='store_true')
    group.add_argument("--monitor-file-path", type=str)
    group.add_argument('--num-stages', type=int, default=None,
                       help='the pipeline stages, this value must be divisible by your GPU num')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    return parser
