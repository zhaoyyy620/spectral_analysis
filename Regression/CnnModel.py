import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=21, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=19, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=17, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn. Linear(38080,1) #8960 ,17920
        self.drop = nn.Dropout(0.2)

    def forward(self,out):
      out = self.conv1(out)
      out = self.conv2(out)
      out = self.conv3(out)
      out = out.view(out.size(0),-1)
      # print(out.size(1))
      out = self.fc(out)
      return out


class AlexNet(nn.Module):
    def __init__(self, num_classes=1, reduction=16):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv2
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv3
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv4
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # conv5
            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # SELayer(256, reduction),
            # nn.LeakyReLU(inplace=True),
        )
        self.reg = nn.Sequential(
            nn.Linear(3840, 1000),  #根据自己数据集修改
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.flatten(start_dim=1)
        out = self.reg(out)
        return out

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,out_C):
        super(Inception,self).__init__()
        self.p1 = nn.Sequential(
            nn.Conv1d(in_c, c1,kernel_size=1,padding=0),
            nn.Conv1d(c1, c1, kernel_size=3, padding=1)
        )
        self.p2 = nn.Sequential(
            nn.Conv1d(in_c, c2,kernel_size=1,padding=0),
            nn.Conv1d(c2, c2, kernel_size=5, padding=2)

        )
        self.p3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=1),
            nn.Conv1d(in_c, c3,kernel_size=3,padding=1),
        )
        self.conv_linear = nn.Conv1d((c1+c2+c3), out_C, 1, 1, 0, bias=True)
        self.short_cut = nn.Sequential()
        if in_c != out_C:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_c, out_C, 1, 1, 0, bias=False),

            )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        out =  torch.cat((p1,p2,p3),dim=1)
        out += self.short_cut(x)
        return out


class DeepSpectra(nn.Module):
    def __init__(self):
        super(DeepSpectra, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=3, padding=0)
        )
        self.Inception = Inception(16, 32, 32, 32, 96)
        self.fc = nn.Sequential(
            nn.Linear(20640, 5000),
            nn.Dropout(0.5),
            nn.Linear(5000, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Inception(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "隐藏层大小必须能被头数整除。"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = hidden_size // num_heads  # 计算每个头部的维度大小

        # 定义查询、键和值的线性变换
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

        # 定义输出的线性变换
        self.dense = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x, batch_size):
        """将最后一个维度分割成 (num_heads, depth)。
        然后转置结果，使其形状变为 (batch_size, num_heads, seq_len, depth)"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.shape[0]

        # 分别对输入x应用线性变换，并分割成多个头
        q = self.split_heads(self.wq(x), batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(self.wk(x), batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(self.wv(x), batch_size)  # (batch_size, num_heads, seq_len, depth)

        # 计算缩放点积注意力，并返回注意力和权重
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)

        # 转置并重塑，准备进行线性变换
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, depth)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.hidden_size)  # (batch_size, seq_len, hidden_size)

        # 应用输出线性变换
        output = self.dense(concat_attention)  # (batch_size, seq_len, hidden_size)

        return output

    def scaled_dot_product_attention(self, q, k, v):
        # 计算查询和键的点积，然后除以根号下的深度，进行缩放
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len, seq_len)

        dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # 在最后一个轴上应用softmax，以获得权重
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len, seq_len)

        # 通过权重加权值，得到输出
        output = torch.matmul(attention_weights, v)  # (..., seq_len, depth)

        return output, attention_weights


class LPCNet(nn.Module):
    def __init__(self):
        super(LPCNet, self).__init__()
        # 卷积层定义
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=25,stride=3,padding=0),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(10, 15, kernel_size=15,stride=3, padding=0),
            nn.BatchNorm1d(15),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(15, 18, kernel_size=10, stride=3,padding=0),
            nn.BatchNorm1d(18),
            nn.ReLU()
        )

        # BiLSTM层定义
        self.hidden_size = 18  # 隐藏层大小
        self.num_layers = 1  # 层数
        self.bilstm = nn.LSTM(input_size=18,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=True)

        # 多头自注意力模块
        self.num_heads = 6  # 定义多头注意力的头数
        self.multihead_attention = MultiHeadAttention(hidden_size=self.hidden_size * 2, num_heads=self.num_heads)

        # 全连接层
        self.fc = nn.Linear(self.hidden_size * 2, 1)  # 输入维度为隐藏层大小的两倍

        # Dropout层
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 为LSTM层准备数据
        x = x.permute(0, 2, 1)  # 重排维度为[batch, seq_len, features]
        # BiLSTM层处理
        lstm_out, _ = self.bilstm(x)
        # 应用多头自注意力机制
        attention_out = self.multihead_attention(lstm_out)
        # 应用Dropout
        attention_out = self.drop(attention_out)

        # 添加平均池化步骤，对每个样本的所有时间步进行平均
        pooled = torch.mean(attention_out, dim=1)

        # 全连接层处理
        out = self.fc(pooled)
        # 返回最后的输出
        return out