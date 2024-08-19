# a transformer from 0 to 1
# author: Jserw
# date: 2024/8/19
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout, is_casual, block_size):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0
        self.is_casual = is_casual
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.att_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)
        self.block_size = block_size
        self.proj_qkv = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(3)])
        self.proj_out = nn.Linear(n_embd, n_embd)
        # 如果自己实现 MHSA，需要一个 causal mask，确保 attention 只能作用在输入序列的左边
        # 此处使用 register_buffer 注册一个 bias 属性
        # bias 是一个上三角矩阵，维度为 1 x 1 x block_size x block_size，block_size 为序列最大长度
        self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size).reshape(1, 1, self.block_size, self.block_size)))

    def forward(self, q,k,v):
        b,n,dim = q.shape
        q,k,v = [self.proj_qkv[i](x) for i,x in zip(range(3),(q,k,v))] # 这个操作有点骚 b,n,dim
        q.reshape(b, n, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # 相当于从词向量的维度去分成head个，操作后每个token的表征向量长度为原来的head分之一
        k.reshape(b, n, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v.reshape(b, n, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # b h n h_e

        att = q @ k.transpose(-1, -2) * math.sqrt(1.0/k.size(-1)) # b h n n
        # 如果有mask的话，需要先计算mask再softmax
        if self.is_casual:
            att = att.masked_fill(self.bias[:,:,:n,:n] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        y = (att @ v).transpose(1,2).reshape(b, n, dim)

        y = self.proj_out(y)
        y = self.proj_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super(MLP, self).__init__()
        self.n_embd = n_embd
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(n_embd*4, n_embd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super(LayerNorm, self).__init__()
        self.weights = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weights.shape, self.weights, self.bias, 1e-5) # 第二个参数给n_dim即可，无需纠结


class EncoderLayer(nn.Module):
    def __init__(self,n_embd, bias, n_head, dropout, is_casual, block_size):
        super(EncoderLayer, self).__init__()
        self.ln1 = LayerNorm(n_embd, bias)
        self.ln2 = LayerNorm(n_embd, bias)
        self.attention = MultiHeadAttention(n_head,n_embd,dropout,False,block_size)
        self.ffn = MLP(n_embd, dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = self.attention(x,x,x) + x
        x = self.ln2(x)
        x = self.ffn(x) + x
        return x


class Encoder(nn.Module):
    def __init__(self,n_layers, n_embd, bias, n_head, dropout, is_casual, block_size):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(n_embd, bias, n_head, dropout, is_casual, block_size) for _ in range(n_layers)])
        self.norm = LayerNorm(n_embd, bias)

    def forward(self, x):
        for layer in self.layers: # NB
            x = layer(x)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,n_embd, bias, n_head, dropout, block_size, is_casual):
        super(DecoderLayer, self).__init__()
        self.ln1 = LayerNorm(n_embd, bias)
        self.mask_att = MultiHeadAttention(n_head, n_embd, dropout, True, block_size)
        self.ln2 = LayerNorm(n_embd, bias)
        self.att = MultiHeadAttention(n_head, n_embd, dropout, False, block_size)
        self.ln3 = LayerNorm(n_embd, bias)
        self.ffn = MLP(n_embd, dropout)

    def forward(self, x, encoder_output):
        x = self.ln1(x)
        x = self.mask_att(x,x,x) + x
        x = self.ln2(x)
        x = self.att(x, encoder_output, encoder_output) + x
        x = self.ln3(x)
        x = self.ffn(x) + x
        return x

class Decoder(nn.Module):
    def __init__(self,n_layers,n_embd, bias, n_head, dropout, block_size, is_casual):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(n_embd, bias, n_head, dropout, block_size, is_casual) for _ in range(n_layers)])
        self.norm = LayerNorm(n_embd, bias)

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x,enc_out)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, block_size, embd_dim, n=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        res = torch.zeros(block_size, embd_dim)
        for pos in range(block_size):
            for i in range(embd_dim):
                if i % 2:
                    fenmu = np.power(n, (i - 1) / embd_dim)
                    res[pos, i] = np.cos(pos / fenmu)
                else:
                    fenmu = np.power(n, i / embd_dim)
                    res[pos, i] = np.sin(pos / fenmu)
        self.pe = res.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)].requires_grad_(False)
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_len, n_layers, bias, n_head, dropout, is_casual, block_size):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.block_size = block_size
        self.vocab_size = vocab_size
        assert self.seq_len <= self.block_size
        self.word_embd = nn.Embedding(vocab_size, embd_dim)
        self.pos_enc = PositionalEncoding(block_size, embd_dim)
        self.encoder = Encoder(n_layers, embd_dim, bias, n_head, dropout, is_casual, block_size)
        self.decoder = Decoder(n_layers, embd_dim, bias, n_head, dropout, block_size, is_casual)
        self.dropout = nn.Dropout(dropout)
        self.linear_head = nn.Linear(embd_dim, vocab_size)

        # 初始化参数(apply函数还蛮厉害的)
        self.apply(self._init_weights)

        # 查看所有参数数量
        print('num of params is: %.2fM' %(self.get_num_params() / 1e6))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, targets=None):
        # input (batch_size, seq_len) targets是目标序列,用于计算loss
        device = input.device
        batch_size, seq_len = input.size()
        input = input.int()
        print('input shape: ', input.shape)
        assert seq_len <= self.block_size # 确保输入序列的长度小于最长长度
        # 首先通过embedding层
        tok_emd = self.word_embd(input) # bs,seq_len --> bs, seq_len, embd_dim
        print('word_embedding shape: ', tok_emd.shape)

        # 加入位置编码
        input_embd = self.pos_enc(tok_emd) # bs, seq_len, embd_dim
        x = self.dropout(input_embd)

        enc_out = self.encoder(x)
        dec_out = self.decoder(x, enc_out)

        if targets is not None:
            # 训练阶段
            logits = self.linear_head(dec_out) # bs, seq_len, vocab_size
            # input(N,C) target(N)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，取需要的最后一个token
            logits = self.linear_head(dec_out[:,[-1],:]) # b, 1, vocab_size
            loss = None

        return logits, loss

    # 配置优化器
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 首先获取所有命名参数
        param_dict = {pn: p for pn,p in self.named_parameters()}
        # 过滤掉不需要更新的参数
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        # 参数根据维度分为两组
        # 维度大于等于2的参数（通常是权重）会应用权重衰减，而维度小于2的参数（通常是偏置和层归一化参数）不会应用权重衰减。
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim()<1]
        optim_groups = [
            {'params':decay_params, 'weight_decay':weight_decay},
            {'params':nodecay_params, 'weight_decay':0.0}
        ]
        # 打印一下参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"应用权重衰减的层数: {len(decay_params)}； 总参数量为：{num_decay_params:,}")
        print(f"不应用权重衰减的层数: {len(nodecay_params)}, 总参数量为：{num_nodecay_params:,}")
        # 检查 torch.optim.AdamW 是否支持融合版本（fused version），这是针对 CUDA 设备优化的版本。如果可用且 device_type 为 'cuda'，则使用融合版本。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # 创建优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"是否使用 fused AdamW: {use_fused}")
        return optimizer

    # 进行推理
    @torch.no_grad()
    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):
            print(_)
            # 推理阶段，输入为 input，维度为 (batch size, sequence length)，max_new_tokens 为最大生成的 token 数量即按序推理 max_new_tokens 次
            # 如果输入序列太长，我们需要将它截断到 block_size
            input_cond = input if input.size(1)<=self.block_size else input[:,-self.block_size:]
            logits,_ = self(input_cond) # (b, 1, vocab_size)
            logits = logits[:,-1,:] # (b, vocab_size)
            probs = F.softmax(logits, dim=-1) # (b, vocab_size)
            input_next = torch.multinomial(probs, num_samples=1) # 按权重采样,(b, 1)
            # 将输出结果拼接到输入序列后面，作为下一次的输入
            input = torch.cat((input, input_next),dim=1) # (b,1) + (b,seq_len) = (b, seq_lem + 1)
        return input


transformer = Transformer(1500,8,10,6,False,8,0.1,True,100)
x = torch.ones(32,10)
y,_ = transformer(x)
out = transformer.generate(x,30)
print(out.shape) # (32,40)
# SUCCESS!!!