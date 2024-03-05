<<<<<<< HEAD
# @modified by: zhijian jiang
import json
import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from tqdm import tqdm
from typing import Union

# class LongRopeDataset(Dataset):
#     def __init__(self, 
#                  data_path, 
#                  tokenizer, 
#                  max_len, 
#                  max_src_len, 
#                  read_nums:Union[int, None]=None):
        
#         self.all_data = []
#         with open(data_path, "r", encoding="utf-8") as fh:
#             if read_nums is None:
#                 read_nums = sum(1 for _ in fh)
#             for i, line in tqdm(enumerate(fh)):
#                 if i<read_nums:
#                     sample = json.loads(line.strip())

#                     meta_prompt = 'You are a helpful long context assitant: \n'
#                     input_text = meta_prompt + 'Q:\n' + sample['input'] 
#                     output_text = 'A:\n' + sample['output']

#                     # Tokenize input and output texts
#                     input_ids = tokenizer.encode(input_text)

#                     output_ids = tokenizer.encode(output_text)
#                     # Truncate input and output if they exceed maximum lengths
#                     if len(input_ids) > max_src_len:
#                         input_ids = input_ids[:max_src_len - 3]

#                     if len(output_ids) > (max_len - len(input_ids) - 3):
#                         output_ids = output_ids[:(max_len - len(input_ids) - 3)]

#                     final_input_ids = input_ids + output_ids
#                     # Pad input_ids with -100 for labels
#                     labels = [tokenizer.pad_id] * len(input_ids) + final_input_ids[len(input_ids):]

#                     assert len(final_input_ids) == len(labels)
#                     assert len(input_ids) <= max_len

#                     self.all_data.append({"input_ids": input_ids, "labels": labels})

#     def __len__(self):
#         return len(self.all_data)

#     def __getitem__(self, idx):
#         instance = self.all_data[idx]
#         return instance

class LongRopeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len,read_nums:Union[int, None]=None):
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            if read_nums is None:
                read_nums = sum(1 for _ in fh)
            for i, line in tqdm(enumerate(fh)):
                if i<read_nums:
                    sample = json.loads(line.strip())

                    meta_prompt = 'You are a helpful long context assitant: \n'
                    input_text = meta_prompt + 'Q:' + sample["input"] + '\n'
                    output_text = 'A:' + sample["output"]
                    # Tokenize input and output texts
                    input_tokens = tokenizer.tokenize(input_text)
                    input_tokens = [token.replace('▁', '') for token in input_tokens]
                    output_tokens = tokenizer.tokenize(output_text)
                    output_tokens = [token.replace('▁', '') for token in output_tokens]

                    # Truncate input and output if they exceed maximum lengths
                    if len(input_tokens) > max_src_len:
                        input_tokens = input_tokens[:max_src_len - 3]

                    if len(output_tokens) > (max_len - len(input_tokens) - 3):
                        output_tokens = output_tokens[:(max_len - len(input_tokens) - 3)]  # Adjust if needed

                    # Combine tokens with special tokens
                    tokens = input_tokens + output_tokens
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    # Pad input_ids with -100 for labels
                    labels = [tokenizer.pad_id] * len(input_tokens) + input_ids[len(input_tokens):]
                    pad_len = max_len - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_id] * pad_len
                    labels = labels + [tokenizer.pad_id] * pad_len

                    assert len(input_ids) == len(labels)
                    assert len(input_ids) <= max_len

                    self.all_data.append({"input_ids": torch.tensor(input_ids).long(), "labels": torch.tensor(labels).long()})
=======
import json
from torch.utils.data import Dataset
from tokenizer import Tokenizer
class GLMPromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False

                src_tokens = tokenizer.tokenize(
                    "[Round {}]\n问：{}\n答：".format(1, sample["instruction"] + sample["input"]))

                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断，但保留“\n答：”内容
                    src_tokens = src_tokens[:max_src_len - 3] + src_tokens[-3:]
                    skip_flag = True

                max_tgt_len = max_len - 3 - len(src_tokens)
                tgt_tokens = tokenizer.tokenize(sample["output"])

                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM需要在输入内容后面增加"[gMASK]"、"<sop>"标记
                tokens = src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = len(src_tokens) + 2
                mask_position = context_length - 1 # 掩码
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance



class LongRopeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len):
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())

                input_text = sample["input"]
                output_text = sample["output"]

                # Tokenize input and output texts
                input_tokens = tokenizer.tokenize(input_text)
                input_tokens = [token.replace('▁', '') for token in input_tokens]
                output_tokens = tokenizer.tokenize(output_text)
                output_tokens = [token.replace('▁', '') for token in output_tokens]

                # Truncate input and output if they exceed maximum lengths
                if len(input_tokens) > max_src_len:
                    input_tokens = input_tokens[:max_src_len - 3]

                if len(output_tokens) > (max_len - len(input_tokens) - 3):
                    output_tokens = output_tokens[:(max_len - len(input_tokens) - 3)]  # Adjust if needed

                # Combine tokens with special tokens
                tokens = input_tokens + ["[SEP]"] + output_tokens
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # Pad input_ids with -100 for labels
                labels = [-100] * len(input_tokens) + input_ids[len(input_tokens):]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len

                self.all_data.append({"input_ids": input_ids, "labels": labels})
>>>>>>> ed45f654d052d9dc8e4e06d4d4d425065babe4af

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        instance = self.all_data[idx]
        return instance

if __name__ == '__main__':
<<<<<<< HEAD
    z = LongRopeDataset('/workspace/gemma/data/LongQLoRA-SFT-Data-39k.jsonl', Tokenizer('/workspace/gemma/data/tokenizer.model'), 1024, 256, 2)
    print(z[0])
=======

    # from glm1.tokenization_chatglm import ChatGLMTokenizer
    # x = ChatGLMTokenizer.from_pretrained(args.model_name_or_path)
    y = GLMPromptDataSet('../data/d2q_0.json', Tokenizer('../data/tokenizer.model'), 1024, 256, True)
    z = LongRopeDataset('../data/LongQLoRA-SFT-Data-39k.jsonl', Tokenizer('../data/tokenizer.model'), 1024, 256)
    print()
>>>>>>> ed45f654d052d9dc8e4e06d4d4d425065babe4af
