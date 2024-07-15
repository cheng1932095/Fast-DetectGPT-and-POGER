# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import datasets
import torch
import random
import argparse
import os
import json
import custom_datasets
from model import load_tokenizer, load_model


def save_data(output_file, args, data):
    # 将参数写入文件
    args_file = f"{output_file}.args.json"  # 参数文件名
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)  # 将参数对象转换为字典并写入JSON文件
        print(f"Args written into {args_file}")  # 打印提示消息，显示参数已写入文件

    # 将数据写入保存文件夹中的JSON文件
    data_file = f"{output_file}.raw_data.json"  # 数据文件名
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)  # 将数据写入JSON文件
        print(f"Raw data written into {data_file}")  # 打印提示消息，显示数据已写入文件


def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data


class DataBuilder:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir)
        self.base_model = None if args.openai_model else load_model(args.base_model_name, args.device, args.cache_dir)

    def _openai_sample(self, prefix):
        def _drop_last_word(text):
            return ' '.join(text.split(' ')[:-1])

        import openai
        assert self.args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = self.args.openai_key

        # 设置 OpenAI API base URL（如果提供）
        if self.args.openai_base is not None:
            openai.api_base = self.args.openai_base

        # 对于非 'pubmed' 数据集，去掉前缀的最后一个单词
        if self.args.dataset != 'pubmed':
            prefix = _drop_last_word(prefix)

        # 设置 OpenAI 请求参数
        kwargs = {"max_tokens": 200}
        if self.args.do_top_p:
            kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            kwargs['temperature'] = self.args.temperature

        if self.args.openai_model == 'davinci':
            # 使用 OpenAI 的 Davinci 模型生成文本
            kwargs["engine"] = self.args.openai_model
            response = openai.Completion.create(prompt=f"{prefix}", **kwargs)
            return prefix + response['choices'][0]['text']
        elif self.args.openai_model in ['gpt-3.5-turbo', 'gpt-4']:
            # 根据不同数据集设定角色和提示语
            roles = {'xsum': 'You are a News writer.',
                     'writing': 'You are a Fiction writer.',
                     'pubmed': 'You are a Technical writer.'}
            prompts = {'xsum': 'Please write an article with about 150 words starting exactly with:',
                       'writing': 'Please write an article with about 150 words starting exactly with:',
                       'pubmed': 'Please answer the question in about 50 words.'}

            # 构建消息对象，用于聊天式生成文本
            messages = [
                {'role': 'system', 'content': roles[self.args.dataset]},
                {'role': 'user', 'content': f'{prompts[self.args.dataset]} {prefix}'},
            ]

            # 设置 OpenAI 请求参数，指定模型和消息
            kwargs["model"] = self.args.openai_model
            kwargs["messages"] = messages
            response = openai.ChatCompletion.create(**kwargs)
            response = response['choices'][0]['message']['content']

            # 检查生成的响应，如果以部分前缀开头，直接返回响应
            if response.startswith(prefix[:20]):
                return response
            return prefix + ' ' + response
        else:
            raise NotImplementedError  # 未实现的模型抛出异常

    def _sample_from_model(self, texts, min_words=55, prompt_tokens=30):
        # 对每个文本进行编码，以列表形式返回token id
        if self.args.dataset == 'pubmed':
            # 如果是 pubmed 数据集，根据特定分隔符截取文本内容
            texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                self.args.device)
        else:
            # 对于其他数据集，只使用前 prompt_tokens 个 token 进行编码
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(
                self.args.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        if self.args.openai_model:
            # 如果使用 OpenAI 模型，将编码后的前缀解码回文本
            prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

            decoded = []
            for idx, prefix in enumerate(prefixes):
                while idx >= len(decoded):
                    try:
                        decoded.append(self._openai_sample(prefix))
                    except Exception as ex:
                        print(ex)
                        print('Wait 10 minutes before retry ...')
                        time.sleep(600)

        else:
            # 启用基础模型的生成模式
            self.base_model.eval()
            decoded = ['' for _ in range(len(texts))]

            # 循环采样，直到生成的样本至少包含 min_words 个单词
            tries = 0
            m = 0
            while m < min_words:
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                    prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                    for prefix, x in zip(prefixes, decoded):
                        if len(x.split()) == m:
                            print(prefix, '=>', x)

                # 设置采样参数
                sampling_kwargs = {}
                if self.args.do_top_p:
                    sampling_kwargs['top_p'] = self.args.top_p
                elif self.args.do_top_k:
                    sampling_kwargs['top_k'] = self.args.top_k
                elif self.args.do_temperature:
                    sampling_kwargs['temperature'] = self.args.temperature

                # 对于 pubmed 数据集，设置最小生成长度为 50，其他情况为 150
                min_length = 50 if self.args.dataset in ['pubmed'] else 150

                # 生成文本
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                                   **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                   eos_token_id=self.base_tokenizer.eos_token_id)
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                m = min(len(x.split()) for x in decoded)
                tries += 1

        return decoded

    def generate_samples(self, raw_data, batch_size):
        # trim to shorter length
        def _trim_to_shorter_length(texta, textb):
            # 截取到较短的长度
            shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
            texta = ' '.join(texta.split(' ')[:shorter_length])
            textb = ' '.join(textb.split(' ')[:shorter_length])
            return texta, textb

        def _truncate_to_substring(text, substring, idx_occurrence):
            # 截取第 idx_occurrence 次出现 substring 之后的内容
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],  # 存储原始文本
            "sampled": [],  # 存储采样后的文本
        }

        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]  # 获取当前批次的原始文本数据
            sampled_text = self._sample_from_model(original_text, min_words=30 if self.args.dataset in [
                'pubmed'] else 55)  # 使用模型对原始文本进行采样

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)  # 对采样文本进行截断处理
                    o = o.replace(custom_datasets.SEPARATOR, ' ')  # 替换原始文本中的特定分隔符

                o, s = _trim_to_shorter_length(o, s)  # 对原始文本和采样文本进行长度截取

                # 添加到数据集中
                data["original"].append(o)
                data["sampled"].append(s)

        return data  # 返回处理后的数据集

def generate_data(args, dataset, key):
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # 加载数据
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)  # 从自定义数据集加载数据
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]  # 加载指定数据集的训练集数据

    # 去除数据中的重复项
    data = list(dict.fromkeys(data))  # 使用字典保留唯一的数据项，与集合不同，这是确定性的操作

    # 去除每个示例的首尾空白
    data = [x.strip() for x in data]

    # 去除每个示例中的换行符
    data = [_strip_newlines(x) for x in data]

    # 尝试仅保留超过250个词的示例（特定数据集）
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.shuffle(data)  # 随机打乱数据顺序
    data = data[:5_000]  # 仅保留前5000个示例，以便后续的处理和标记化节省时间

    # 仅保留每个示例的token数不超过512个（基础模型要求）
    data_builder = DataBuilder(args)
    tokenized_data = data_builder.base_tokenizer(data)  # 使用基础的分词器对数据进行标记化
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # 打印关于剩余数据的统计信息
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data_builder.generate_samples(data[:args.n_samples], batch_size=args.batch_size)  # 生成样本数据并返回

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_gpt3/data/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--openai_base', type=str, default="https://api.xiaoai.plus/v1")
    parser.add_argument('--openai_key', type=str, default="sk-XRKhE2CQcEepJvNHB8A6990aD82a497dBe08D71cD135C176")
    parser.add_argument('--openai_model', type=str, default="gpt-4")  # davinci, gpt-3.5-turbo, gpt-4
    parser.add_argument('--base_model_name', type=str, default="gpt2")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document'}
    data = generate_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)

    save_data(args.output_file, args, data)
