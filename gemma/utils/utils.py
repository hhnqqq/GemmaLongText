import os
import time
import json
import torch
import configparser

"""with 环境用法：进入with语句之后，类中__enter__对应的返回值将被赋值给as后面的变量名
with之后的代码执行"""
class Timer(object):
    '''用法：

    with Timer() as timer:
        your code
    print(f"cost:{timer.time_cost}")
    print(f"start:{timer.start}")
    print(f"end:{timer.end}")
    '''
    def __init__(self, start=None, n_round=2):
        self.round = n_round
        self.start = round(start if start is not None else time.time(),self.round)
    def __enter__(self):
        return self  
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = round(time.time(),self.round)
        self.time_cost = round(self.stop - self.start, self.round)
        self.formated_time_cost = f"{int(self.time_cost//60)}min {self.time_cost%60}s"
        return exc_type is None
    

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def load_ckpt(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def read_config(file_path, encoding='utf-8'):
    """读取配置文件"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding=encoding) as f:
            config = json.load(f)
    elif file_path.endswith('.ini'):
        config = configparser.ConfigParser()
        config.read(file_path)
    else:
        raise ValueError("无法读取不支持的文件格式")
    return config

def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('>>> 目录不存在，已新建对应目录')
