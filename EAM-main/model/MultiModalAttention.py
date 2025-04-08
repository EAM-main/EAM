import torch
import torch.nn as nn
import math

class MultiModalAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, use_pos_enc=False, max_len=512):
        super(MultiModalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.fc = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, max_len)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
        nn.init.constant_(self.linear_q.bias, 0.)
        nn.init.constant_(self.linear_k.bias, 0.)
        nn.init.constant_(self.linear_v.bias, 0.)
        nn.init.constant_(self.fc.bias, 0.)
        
    def forward(self, x, mask=None, return_attn=False):

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        

        if self.use_pos_enc:
            x = self.pos_enc(x)
        
        residual = x
        

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        

        if mask is not None:

            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, -1e10)
        

        attention = torch.softmax(energy, dim=-1)
        attention = self.attn_dropout(attention)

        x = torch.matmul(attention, v)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, -1)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        
        if return_attn:
            return x, attention
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
