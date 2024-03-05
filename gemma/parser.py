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
    group.add_argument('--batch-size-per-gpu', type=int, default=4,
                       help='batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bf16 mode')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='run optimizer after every gradient-accumulation-steps backwards.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--device', type=str, default='cpu',
                       help='the device to load the model')
    group.add_argument('--num-stages', type=int, default=None,
                       help='the pipeline stages')
    group.add_argument('--read-nums', type=int, default=None,
                       help='the number of data to read')
    group.add_argument('--max-len', type=int, default=None,
                       help='max len of tokens')
    group.add_argument('--max-src-len', type=int, default=None,
                       help='max len of input tokens')
    group.add_argument('--seed', type=int, default=None,
                       help='random seed')
    group.add_argument('--rope-theta', default=1000.0,
                       help='repe theta')
    group.add_argument('--show-loss-step', type=int, default=1)
    group.add_argument('--variant', type=str, default='2b')
    group.add_argument('--train_pi', action='store_true')
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

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    return parser
