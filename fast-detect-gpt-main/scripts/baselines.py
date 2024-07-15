# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics

def get_likelihood(logits, labels):
    # 断言确保logits和labels的batch size为1
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # 将logits和labels视图重新调整形状为一维张量
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    # 计算log_softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # 根据labels获取对应位置的log概率
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # 返回平均log似然值
    return log_likelihood.mean().item()


def get_rank(logits, labels):
    # 断言确保logits和labels的batch size为1
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # 获取每个标签在模型似然排序中的排名
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # 确保每个时间步中有且仅有一个匹配项
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # 转换为从1开始的排名
    return -ranks.mean().item()  # 返回负平均排名值


def get_logrank(logits, labels):
    # 断言确保logits和labels的batch size为1
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # 获取每个标签在模型似然排序中的排名
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # 确保每个时间步中有且仅有一个匹配项
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # 转换为从1开始的排名
    ranks = torch.log(ranks)  # 计算排名的对数值
    return -ranks.mean().item()  # 返回负平均对数排名值


def get_entropy(logits, labels):
    # 断言确保logits和labels的batch size为1
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # 计算softmax后的熵
    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)

    # 对每个样本计算熵并求和
    entropy = -entropy.sum(-1)

    # 返回平均熵值
    return entropy.mean().item()


def experiment(args):
    # 加载模型
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)  # 加载评分模型的分词器
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)  # 加载评分模型
    scoring_model.eval()  # 设置评分模型为评估模式

    # 加载数据
    data = load_data(args.dataset_file)  # 加载数据集文件
    n_samples = len(data["sampled"])  # 获取样本数量

    # 评估指标函数
    criterion_fns = {
        'likelihood': get_likelihood,  # 获取似然函数的评估指标
        'rank': get_rank,  # 获取排名函数的评估指标
        'logrank': get_logrank,  # 获取对数排名函数的评估指标
        'entropy': get_entropy  # 获取熵函数的评估指标
    }

    # 对每个评估指标函数进行循环计算
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]  # 获取当前评估指标函数
        torch.manual_seed(args.seed)  # 设置PyTorch随机种子
        np.random.seed(args.seed)  # 设置NumPy随机种子
        eval_results = []  # 存储评估结果的列表
        # 对每个样本进行评估
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):  # 使用进度条显示当前评估指标的计算进度
            original_text = data["original"][idx]  # 获取原始文本
            sampled_text = data["sampled"][idx]  # 获取采样文本
            # 处理原始文本
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True,
                                          return_token_type_ids=False).to(args.device)  # 对原始文本进行分词和张量化，并移到指定设备
            labels = tokenized.input_ids[:, 1:]  # 获取输入标签（移除开头的特殊标记）
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]  # 使用评分模型生成logits，去除末尾的特殊标记
                original_crit = criterion_fn(logits, labels)  # 计算原始文本的评估指标值
            # 处理采样文本
            tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                          return_token_type_ids=False).to(args.device)  # 对采样文本进行相同的处理
            labels = tokenized.input_ids[:, 1:]  # 获取输入标签
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]  # 使用评分模型生成logits
                sampled_crit = criterion_fn(logits, labels)  # 计算采样文本的评估指标值
            # 存储评估结果
            eval_results.append({
                "original": original_text,
                "original_crit": original_crit,
                "sampled": sampled_text,
                "sampled_crit": sampled_crit
            })

        # 计算真实和采样文本的预测分数
        predictions = {'real': [x["original_crit"] for x in eval_results],  # 收集所有原始文本的评估指标值
                       'samples': [x["sampled_crit"] for x in eval_results]}  # 收集所有采样文本的评估指标值
        # 计算ROC曲线下面积和Precision-Recall曲线下面积
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])  # 获取ROC曲线的FPR、TPR和AUC值
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions[
            'samples'])  # 获取Precision-Recall曲线的Precision、Recall和AUC值
        # 打印结果
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        # 记录结果
        results_file = f'{args.output_file}.{name}.json'  # 定义结果文件名
        results = {
            'name': f'{name}_threshold',  # 结果名称
            'info': {'n_samples': n_samples},  # 记录样本数量信息
            'predictions': predictions,  # 存储预测结果
            'raw_results': eval_results,  # 存储原始评估结果
            'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},  # 存储ROC曲线指标
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},  # 存储Precision-Recall曲线指标
            'loss': 1 - pr_auc  # 计算损失值（1 - PR AUC）
        }
        with open(results_file, 'w') as fout:
            json.dump(results, fout)  # 将结果写入JSON文件
            print(f'Results written into {results_file}')  # 打印结果写入的文件路径


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
