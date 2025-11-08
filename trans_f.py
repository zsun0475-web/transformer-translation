import torch
import torch.nn as nn
import math #数学运算库

# 定义自注意力
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):#丢弃率，神经元随机失活，防止过拟合
        super().__init__()
        self.dropout = nn.Dropout(dropout) #对10%神经元随机失活
        self.softmax = nn.Softmax(dim=-1) #将得分转换为概率分布，最后一维进行

    def forward(self,Q,K,V,mask=None):
        # 输入向量x:batch,seq_len,d_model
        # batch:一次送入的样本数；seq_len:一个样本中token数（词数）；
        # d_model:embedding向量的维度（隐藏维度），一般transformer为512
        # Q，query向量 （维度：batch, heads(8), seq_len_q, d_k）
        # K，key向量 （维度：batch, heads(8), seq_len_k, d_k）
        # V，value向量 （维度：batch, heads(8), seq_len_v, d_v）
        # mask 哪些位置需要忽略（不看之后的信息）
        d_k = Q.size(-1) #归一化，取Q的最后一维，不取K的，对每个query进行缩放
        # K.transpose(-2,-1)是K的转置,对K的最后两个维度交换
        # batch, heads, seq_len_q, d_k; batch, heads, d_k, seq_len_k->batch, heads, seq_len_q，seq_len_k
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(d_k) #缩放，梯度更稳定
        # 如果提供了mask, 通过mask==0来找到屏蔽位置，asked_fill会将这些值改为负无穷
        # 经过softmax后这些位置值会为0
        # mask==1表示当前位置可见
        if mask is not None:
            scores = scores.masked_fill(mask==0,float('-inf'))
        # batch, heads, seq_len_q，seq_len_k， 对key进行，得到注意力权重矩阵，每个query的key做softmax，和为1
        attn = self.softmax(scores)
        attn = self.dropout(attn) #对注意力权重dropout, 防止过拟合
        # batch, heads, seq_len_q，seq_len_k; V:batch, heads, seq_len_v, d_v->batch, heads, seq_len_q,d_v
        out = torch.matmul(attn,V)
        return out,attn
    
# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads,dropout=0.1):
        super().__init__()
        # d_model 需要被 n_heads整除结果为64
        # d_model embedding的维度512
        # n_heads头数为8
        assert d_model % n_heads == 0 #求余
        self.d_k = d_model // n_heads #每个头的维度
        self.n_heads = n_heads

        #将输入映射到Q K V 三个向量,通过线性映射让模型具有学习能力
        self.W_q = nn.Linear(d_model,d_model)# query的线性映射，维度不需要改变方便后续多头拆分。
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.fc = nn.Linear(d_model,d_model) #多头拼接后再映射回原来的d_model ，模型融合不同头信息

        self.attention = SelfAttention(dropout)# 使用定义好的selfatt
        self.dropout = nn.Dropout(dropout)# 防止过拟合
        self.norm = nn.LayerNorm(d_model) #用于残差后的归一化

    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0) #获取barch的大小
        # q 的维度 batch,seq_len,d_model-> batch,seq_len,self.n_heads,self.d_k-> batch_size,self.n_heads,seq_len,self.d_k
        # 每个头来独立处理整个序列，方便注意力权重计算
        Q = self.W_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)

        # 计算注意力
        out, attn =self.attention(Q,K,V,mask)
        # out.transpose(1,2):out, attn = self.attention(Q,K,V,mask) # attn为注意力权重，out 为注意力加权后的值
        # batch, heads, seq_len_q,d_v->batch, seq_len_q,heads,d_v-> batch,seq_len,d_model
        # contiguous tensor在内存中连续存储， 避免view时候产生报错
        # 多头拼接
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)
        out = self.fc(out) # 让输入与输出一致, 方便残差连接
        out = self.dropout(out) #在训练阶段随机丢弃一部分神经元, 避免过拟合
        return self.norm(out+q), attn #返回输出和注意力权重
    
class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model,d_ff)# 输入维度为d_model,输出为d_ff,512->2048，使模型学到更丰富的特征
        self.fc2 = nn.Linear(d_ff,d_model)# 保证第二个线性层输出维度=第一个线性层的输入维度，为了后续做残差连接
        self.dropout = nn.Dropout(dropout)# 做随即丢弃防止过拟合
        self.norm = nn.LayerNorm(d_model)# layernorm对最后一维进行归一化
    def forward(self,x):
        # x 形状为 batch,seq_len,d_model
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))#先经过第一个线性层，在经过relu,在经过dropout,再经过第二个线性层
        return self.norm(out+x) #残差连接目的，保留输入低阶信息， 避免训练时信息丢失。 
    # 先经过残差连接，再经过层归一化（模型训练更稳定，加快模型收敛）

class EncoderLayer(nn.Module):
    def __init__(self, d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        # 多头注意力机制，输入为src 实现序列内部的信息交互，每个tokem都可以看到序列里的其他词，学习到上下文依赖
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        # 对每个位置向量独立进行非线性变换，提升模型表达能力（前馈）
        self.ffn = FeedForward(d_model,d_ff,dropout)

    def forward(self,src,src_mask=None):
        # src 输入序列张量 形状batch.seq_len,d_model
        # src_mask 用来屏蔽padding位置，避免模型关注无效token(encoder)
        # Q K V = src 对输入序列本身进行自注意力计算
        out,_ = self.self_attn(src,src,src,src_mask)
        # 经过前馈神经网络，每个位置的token都会单独通过两层线性映射和激活函数，提升模型的表达能力
        out = self.ffn(out)
        # 返回编码后的结果
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model,n_heads,d_ff,dropout=0.1 ):
        super().__init__()
        # mask多头注意力机制
        # 输入为tgt(目标序列)，在翻译任务中已经生成的前几个单词。
        # 计算目标序列内部的自注意力，通过mask遮挡未来token
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        # 交叉注意力，和encoder做交互
        # 输入 Q=当前解码器的输出，K=V=来自编码器的信息
        # 为了将目标序列与原序列对其
        #self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.ffn = FeedForward(d_model,d_ff,dropout)# 为了提升模型的表达能力

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        # tgt 目标序列；memory:编码器输出（原序列表示）
        # tgt_mask:用来屏蔽未来token; memory_mask: Pad做掩码，防止没用位置
        # 目标序列内部的自注意力，未来位置被mask
        out,_ = self.self_attn(tgt,tgt,tgt,tgt_mask)
        # 将目标序列和原序列进行交互，Q解码器当前的输出out,K=V=memory
        #out,_ = self.cross_attn(out,memory,memory,memory_mask)
        out = self.ffn(out)
        return out
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出映射到词表大小
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None, use_pos_encoding=True):
        # tgt: [batch_size, seq_len]（输入token ID）
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)  # 缩放嵌入
        out = self.pos_encoding(out, use_pos_encoding)  # 位置编码（支持关闭）
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, tgt_mask)
        out = self.fc_out(out)  # 输出: [batch_size, seq_len, vocab_size]
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len=5000):
        super().__init__()
        # d_model每个词向量的维度； max_len:句子的最大长度
        # 初始化位置编码矩阵，形状为max_len,d_model
        pe = torch.zeros(max_len,d_model)

        # 定义记录每个token位置的索引，0-max_len-1
        # [max_len,1]方便后续与缩放因子进行相乘
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        # div_term 每个维度缩放因子，torch.arrange(0,d_model,2生成偶数维度索引->2i
        # 整个公式 (2i/d_model)*(-ln(10000.0))
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        # 每个token位置索引*每个维度缩放因子，再套上sin
        pe[:,0::2] = torch.sin(position*div_term)
        # 每个token位置索引*每个维度缩放因子，再套上cos,奇数维度位置编码值
        pe[:,1::2] = torch.cos(position*div_term)
        # 增加在第0维，增加batch维度，1，max_len,d_model,方便后续与输入embedding 进行相加
        pe = pe.unsqueeze(0)
        # 注册为buffer,把位置编码pe存在模型里面，不参与训练,但随模型保存/加载
        self.register_buffer('pe',pe)
    def forward(self,x,use_pos_encoding=True):
        if not use_pos_encoding:
            return x #消融实验
        # x:输入的embedding形状 batch,seq_len,d_model
        seq_len = x.size(1)
        # 每个token的embedding加上对应位置编码
        # self.pe[:,:seq_len,:]取前seq_len长度，形状变成1，seq_len,d_model,与x对齐
        # x+self.pe[:,:seq_len,:]:batch,seq_len,d_model(embedding加上位置编码，可以知道token的位置)
        return x+self.pe[:,:seq_len,:]

class Encoder(nn.Module):
    def __init__(self, vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        # 词嵌入，vocab_size:词表大小，包含了不同的token总数
        # 将输入的token ID(对原始文本进行分词得到词表，对应不同ID)转换成连续向量，维度为d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
        # 位置编码加入序列中token的位置信息
        self.pos_encoding = PositionalEncoding(d_model,max_len)

        # 构建编码器堆叠结构
        # 堆叠num_layers个encoder
        # nn.ModuleList为网络层准备的列表，用来存放多个子模块
        # 列表生成/推导式
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)
        ])
    def forward(self,src,src_mask=None):
        # 将输入token ID转换成embedding向量
        # 输出 shape batch,seq_len,d_model
        # 乘sqrt(d_model),缩放，后续注意力计算更稳定
        out = self.embedding(src)*math.sqrt(self.embedding.embedding_dim)
        # 经过位置编码
        out = self.pos_encoding(out)
        # 逐层经过encoderlayer
        for layer in self.layers:
            out = layer(out,src_mask)# self_attn+ffn
        return out # 返回编码后的输出， batch,seq_len,d_model
    
class Decoder(nn.Module):
    def __init__(self, vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super().__init__()
        # 将目标序列的token ID转化为向量，维度：d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model,max_len)
        # 定义解码器列表
        self.layers = nn.ModuleList([
            DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)
        ])
        # 输出投影层，将decoder输出映射回原词汇表的大小，得到每个token的预测分布
        self.fc_out = nn.Linear(d_model,vocab_size)

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        # tgt 目标序列， memory:编码器输出，上下文信息
        out = self.embedding(tgt)*math.sqrt(self.embedding.embedding_dim)
        out = self.pos_encoding(out)
        # 逐层经过decoderlayer
        for layer in self.layers:
            out = layer(out,memory, tgt_mask,memory_mask)
        # 将解码器最后一层输出的隐藏向量映射回原词汇表的维度，得到每个token的预测向量
        return self.fc_out(out)
    
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab, # 原语言词表大小
                 tgt_vocab, # 目标语言词表大小
                 d_model=512,# embedding向量维度
                 n_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000): 
        super().__init__()
        # 编码器：原语言token编码为上下文表示
        self.encoder = Encoder(
            src_vocab,d_model,n_heads,num_encoder_layers,d_ff,dropout, max_len
        )
        # 解码器： 根据编码器的输出和目标语言输入生成预测
        self.decoder = Decoder(
            tgt_vocab,d_model,n_heads,num_decoder_layers,d_ff,dropout, max_len
        )
    def forward (self, src,tgt,src_mask=None,tgt_mask=None,memory_mask=None):
        # 编码器前向传播
        memory = self.encoder(src,src_mask)
        out = self.decoder(tgt,memory, tgt_mask,memory_mask)
        # 返回transformer输出 batch, seq_len_tgt,tgt_vocab
        return out

def generate_mask(size):
    # torch.triu（）生成上三角，不含对角线
    mask = torch.triu(torch.ones(size,size),diagonal=1).bool()
    return mask==0 # True可见，False屏蔽（取反，屏蔽上三角内容）

src_vocab = 10000 #原语言词表大小
tgt_vocab = 10000 #目标语言词表大小
# 初始化模型
model = Transformer(src_vocab,tgt_vocab)
src = torch.randint(0,src_vocab,(32,10)) #原序列batch=32，src_len=10 每个元素时token ID
tgt = torch.randint(0,tgt_vocab,(32,20))
# (tgt.size(1))取目标序列长度，
tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)
out = model(src,tgt,tgt_mask=tgt_mask) #前向传播
# 每个目标token对应词表中每个词的预测概率
print(out.shape) # batch,tgt_len,tgt_vocab

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset #Hugging Face加载
import numpy as np
import os

class TinyShakespeareDataset(Dataset):
    def __init__(self,data,seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.chars = sorted(list(set(data)))  # 字符级词表（去重排序）
        self.vocab_size = len(self.chars)
        # 字符→ID映射（用于编码）
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        # ID→字符映射（用于解码生成文本）
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        # 将整个文本编码为ID序列
        self.data_ids = torch.tensor([self.char2idx[c] for c in data], dtype=torch.long)

    def __len__(self):
        # 数据集长度 = 总ID数 - seq_len（避免越界）
        return len(self.data_ids) - self.seq_len

    def __getitem__(self, idx):
        # 输入x: [idx, idx+seq_len]，标签y: [idx+1, idx+seq_len+1]（预测下一个字符）
        x = self.data_ids[idx:idx + self.seq_len]
        y = self.data_ids[idx + 1:idx + self.seq_len + 1]
        return x, y
def load_local_tiny_shakespeare(seq_len=32, batch_size=8, data_path="data/tiny_shakespeare/input.txt"):
    # 步骤1：读取本地input.txt文件
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件未找到，请确认路径：{data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()  # 读取完整文本（约1MB）
    
    # 步骤2：划分训练集/验证集（9:1划分）
    train_ratio = 0.9
    train_text = full_text[:int(len(full_text) * train_ratio)]  # 训练集（90%）
    val_text = full_text[int(len(full_text) * train_ratio):]    # 验证集（10%）
    
    # 步骤3：构建Dataset和DataLoader
    train_dataset = TinyShakespeareDataset(train_text, seq_len)
    val_dataset = TinyShakespeareDataset(val_text, seq_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True  # 训练集打乱（文档1-52预处理要求）
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True  # 验证集不打乱（文档1-72结果稳定要求）
    )
    
    return train_loader, val_loader, train_dataset.vocab_size, train_dataset.idx2char

if __name__ == "__main__":
    # 调用加载函数
    train_loader, val_loader, vocab_size, idx2char = load_local_tiny_shakespeare(
        seq_len=32, batch_size=32
    )
    # 验证数据格式（确保适配模型输入）
    x, y = next(iter(train_loader))
    print(f"训练样本输入形状：{x.shape} → (batch_size, seq_len)")  # torch.Size([32, 32])
    print(f"训练样本标签形状：{y.shape} → (batch_size, seq_len)")    # torch.Size([32, 32])
    print(f"字符级词表大小：{vocab_size} → 适配模型embedding层输入")  # 约65（大小写+标点）

import matplotlib.pyplot as plt
from datetime import datetime

def generate_padding_mask(x, pad_idx=0, device=None):
    if device is None:
        device = x.device
    return (x != pad_idx).unsqueeze(1).unsqueeze(2).to(device)  # [batch, 1, 1, seq_len]

def save_model(model, optimizer, epoch, train_loss, val_loss, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    print(f"Model loaded from {load_path} (epoch {epoch})")
    return model, optimizer, epoch, train_loss, val_loss

def plot_training_curves(train_losses, val_losses, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Transformer LM Training & Validation Loss Curves (Tiny Shakespeare)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 保存图片（带时间戳，避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"loss_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {save_path}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse
import tqdm
import data
from src.trans_f import DecoderOnlyTransformer
from src.trans_f import (
    generate_future_mask, set_seed, save_model, plot_training_curves
)
