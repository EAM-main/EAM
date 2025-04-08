import torch
import torch.nn as nn
import math


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1, use_pos_enc=False, max_len=512):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc_q = PositionalEncoding(d_model, max_len)
            self.pos_enc_k = PositionalEncoding(d_model, max_len)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)

        nn.init.constant_(self.q_linear.bias, 0.)
        nn.init.constant_(self.k_linear.bias, 0.)
        nn.init.constant_(self.v_linear.bias, 0.)
        nn.init.constant_(self.fc_q.bias, 0.)
        nn.init.constant_(self.fc_k.bias, 0.)

    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.shape[0]
        q_len, k_len, v_len = query.shape[1], key.shape[1], value.shape[1]

        if self.use_pos_enc:
            query = self.pos_enc_q(query)
            key = self.pos_enc_k(key)

        residual_q = query
        residual_k = key

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(batch_size, q_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, k_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, v_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, -1e10)

        attention_q2k = torch.softmax(energy, dim=-1)
        attention_q2k = self.attn_dropout(attention_q2k)

        x_q = torch.matmul(attention_q2k, v)

        x_q = x_q.permute(0, 2, 1, 3).contiguous()
        x_q = x_q.view(batch_size, q_len, -1)

        energy_k = torch.matmul(k, q.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            mask_t = mask.permute(0, 1, 3, 2)
            energy_k = energy_k.masked_fill(mask_t == 0, -1e10)

        attention_k2q = torch.softmax(energy_k, dim=-1)
        attention_k2q = self.attn_dropout(attention_k2q)
        x_k = torch.matmul(attention_k2q, v)
        x_k = x_k.permute(0, 2, 1, 3).contiguous()
        x_k = x_k.view(batch_size, k_len, -1)

        q_out = self.fc_q(x_q)
        q_out = self.dropout(q_out)
        query_out = self.norm_q(residual_q + q_out)

        k_out = self.fc_k(x_k)
        k_out = self.dropout(k_out)
        key_out = self.norm_k(residual_k + k_out)

        if return_attn:
            return query_out, None, key_out, (attention_q2k, attention_k2q)

        return query_out, None, key_out


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
