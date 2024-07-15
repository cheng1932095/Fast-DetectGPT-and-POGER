import os
import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, optim
from transformers import RobertaModel
from typing import List, Tuple

from utils.functions import MLP
from utils.trainer import Trainer as TrainerBase

class Attention(nn.Module):
    """
    计算缩放点积注意力（Scaled Dot Product Attention）
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        # 如果存在掩码，将掩码应用到注意力分数上
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 对注意力分数进行softmax操作，得到注意力权重
        p_attn = F.softmax(scores, dim=-1)

        # 如果指定了dropout，应用dropout到注意力权重上
        if dropout is not None:
            p_attn = dropout(p_attn)

        # 使用注意力权重加权求和value的加权和，并返回
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
    """
    多头注意力机制模块，接受模型大小和头数作为参数。
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # 假设 d_v 总是等于 d_k
        self.d_k = d_model // h
        self.h = h

        # 定义线性变换层，共三个，分别用于变换 query、key 和 value
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        # 输出线性变换层，将多头注意力的结果映射回原始维度
        self.output_linear = torch.nn.Linear(d_model, d_model)
        # 注意力计算的实例
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)

        # 1) 执行线性投影，将输入变换到 h x d_k 的维度
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) 在批处理中应用注意力机制
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用 view 进行“连接”，然后应用最终的线性变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        # 初始化多头注意力机制
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
        # 可选的输出层（线性变换）
        # self.out_layer = torch.nn.Linear(input_size, output_size)

    def forward(self, inputs, query, mask=None):
        # 如果存在掩码，调整掩码的形状以便多头注意力机制使用
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        # 调用多头注意力机制，传入查询(query)、值(inputs)和键(inputs)，以及可选的掩码(mask)
        feature, attn = self.attention(query=query,
                                       value=inputs,
                                       key=inputs,
                                       mask=mask
                                       )
        # 返回处理后的特征和注意力矩阵
        return feature, attn

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2  # 计算填充大小
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),  # 一维卷积层
                nn.Dropout(conv_dropout),  # Dropout层
                nn.ReLU()  # ReLU激活函数
            )

        in_d = 1  # 输入通道数
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))  # 添加卷积块到ModuleList中
            in_d = dim  # 更新输入通道数

    def forward(self, x):
        # x = x.unsqueeze(1)  # 如果输入x的维度不是[B, C, L]，可以进行unsqueeze操作
        for conv in self.conv_layers:
            x = conv(x)  # 逐层应用卷积块
        return x  # 返回处理后的特征张量

class Model(nn.Module):

    def __init__(self, nfeat, nclasses, dropout=0.2, k=10):
        super(Model, self).__init__()
        self.nfeat = nfeat

        # 定义卷积特征提取层
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        embedding_size = nfeat * 64

        # 定义Transformer编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # 定义位置编码
        seq_len = k
        self.position_encoding = torch.zeros((seq_len, embedding_size))

        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000 ** ((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000 ** ((2 * (i + 1)) / embedding_size))))


        # 归一化层和dropout层
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(0.1)

        # 定义MLP作为特征降维器
        self.reducer = MLP(768, [384], embedding_size, dropout)

        # 定义自注意力机制模型
        self.cross_attention_context = SelfAttentionFeatureExtract(1, embedding_size)
        self.cross_attention_prob = SelfAttentionFeatureExtract(1, embedding_size)

        # 定义分类器MLP
        self.classifier = MLP(embedding_size * 2, [128, 32], nclasses, dropout)

    def conv_feat_extract(self, x):
        # 定义卷积特征提取函数
        out = self.conv(x)  # 使用卷积层提取特征
        out = out.transpose(1, 2)  # 转置特征维度，使得特征维度在第2个位置
        return out

    def forward(self, prob_feature, sem_feature, target_roberta_idx):
        # 前向传播函数
        # prob_feature的形状：(batch_size, nfeat, seq_len)
        # sem_feature的形状：(batch_size, 10, 768)
        # target_roberta_idx的形状：(batch_size, 10)

        # 提取目标位置的语义特征
        context_feature = sem_feature.gather(1, target_roberta_idx.unsqueeze(-1).expand(-1, -1, sem_feature.shape[-1]))
        # context_feature的形状：(batch_size, 10, 768)

        # 减少语义特征维度
        context_feature = self.reducer(context_feature)
        # context_feature的形状：(batch_size, 10, embedding_size)

        # 对prob_feature进行处理
        prob_feature = prob_feature.transpose(1, 2)  # 转置特征维度以适应卷积操作
        # prob_feature的形状：(batch_size, seq_len, nfeat)

        # 使用卷积特征提取函数对prob_feature进行特征提取
        prob_feature = torch.cat([self.conv_feat_extract(prob_feature[:, i:i + 1, :]) for i in range(self.nfeat)],
                                 dim=2)
        # prob_feature的形状：(batch_size, 10, embedding_size)

        # 添加位置编码并处理
        prob_feature = prob_feature + self.position_encoding.cuda()
        prob_feature = self.norm(prob_feature)  # 归一化
        prob_feature = self.encoder(prob_feature)  # Transformer编码器处理
        prob_feature = self.dropout(prob_feature)  # dropout层处理

        # 重新加权prob_feature
        prob_feature, _ = self.cross_attention_prob(prob_feature, context_feature)
        # prob_feature的形状：(batch_size, 10, embedding_size)

        # 重新加权context_feature
        context_feature, _ = self.cross_attention_context(context_feature, prob_feature)
        # context_feature的形状：(batch_size, 10, embedding_size)

        # 拼接处理后的特征
        merged = torch.cat([prob_feature, context_feature], dim=-1)
        # merged的形状：(batch_size, 10, embedding_size * 2)

        # 分类器处理
        merged = self.classifier(merged)
        # merged的形状：(batch_size, 10, nclasses)

        # 对最后一维取平均，得到最终输出
        output = merged.mean(dim=1)
        # output的形状：(batch_size, nclasses)

        return output


class Trainer(TrainerBase):
    def __init__(self, device, pretrain_model, train_dataloader, test_dataloader, epoch, lr, model_save_path, n_classes,
                 n_feat, k):
        super(Trainer, self).__init__(device, pretrain_model, train_dataloader, test_dataloader, epoch, lr,
                                      model_save_path, n_classes)

        # 初始化预训练模型和保存路径
        self.pretrain = RobertaModel.from_pretrained(pretrain_model).to(device)
        self.model_save_path = model_save_path

        # 初始化模型、损失函数和优化器
        self.model = Model(nfeat=n_feat, nclasses=n_classes, k=k).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_loss(self, batch):
        # 将数据移到设备上
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # 使用预训练模型提取语义特征，并且分离计算图
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()

        # 将概率特征移到设备上
        prob_feature = batch['est_prob'].to(self.device)

        # 将目标Robert索引移到设备上
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)

        # 将标签移到设备上
        label = batch['label'].to(self.device)

        # 前向传播
        output = self.model(prob_feature, sem_feature, target_roberta_idx)

        # 计算损失
        loss = self.criterion(output, label)
        return loss

    def get_output(self, batch):
        # 将数据移到设备上
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # 使用预训练模型提取语义特征，并且分离计算图
        sem_feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()

        # 将概率特征移到设备上
        prob_feature = batch['est_prob'].to(self.device)

        # 将目标Robert索引移到设备上
        target_roberta_idx = batch['target_roberta_idx'].to(self.device)

        # 禁止梯度计算，进行推理
        with torch.no_grad():
            output = self.model(prob_feature, sem_feature, target_roberta_idx)

        return output
