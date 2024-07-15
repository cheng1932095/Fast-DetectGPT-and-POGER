# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # 如果本地模型存在，则使用本地模型
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)  # 使用本地路径加载预训练模型，并传入额外参数kwargs
    # 否则，从远程下载预训练模型
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)  # 下载指定模型名称的预训练模型，传入额外参数kwargs，并指定缓存目录为cache_dir


# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_model(model_name, device, cache_dir):
    # 获取完整的模型名称
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')  # 打印正在加载的模型名称

    model_kwargs = {}  # 初始化模型加载的参数字典

    # 如果模型在float16_models列表中，设置torch_dtype为torch.float16
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))

    # 如果模型名称包含'gpt-j'，设置revision为'float16'
    if 'gpt-j' in model_name:
        model_kwargs.update(dict(revision='float16'))

    # 调用from_pretrained方法加载预训练模型
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)

    # 将加载的模型移动到指定的设备（GPU）
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')  # 打印模型加载完成所需的时间

    return model  # 返回加载并移动到指定设备后的模型对象


def load_tokenizer(model_name, for_dataset, cache_dir):
    # 获取完整的模型名称
    model_fullname = get_model_fullname(model_name)

    optional_tok_kwargs = {}  # 初始化可选的分词器参数字典

    # 如果模型名称中包含"facebook/opt-"，使用非快速分词器（tokenizer）
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False

    # 根据数据集类型设置填充的位置
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'  # 对于'pubmed'数据集，设置填充在左侧
    else:
        optional_tok_kwargs['padding_side'] = 'right'  # 默认情况下，设置填充在右侧

    # 使用from_pretrained方法加载自动选择的分词器（AutoTokenizer）
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)

    # 如果分词器的pad_token_id为None，则设置为eos_token_id
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

        # 如果模型名称中包含'13b'，将pad_token_id设置为0
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0

    return base_tokenizer  # 返回加载并配置后的分词器对象


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="bloom-7b1")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    load_model(args.model_name, 'cpu', args.cache_dir)
