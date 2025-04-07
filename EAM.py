import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trm import *
from model.attention import *
import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from collections import defaultdict, deque
import random


# 事件记忆库类，用于事件对比学习
class EventMemoryBank:
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.bank = defaultdict(lambda: {'content_e': deque(), 'content_t': deque()})
        self.tau = 0.8

    def push(self, event_ords, content_e, content_t):
        for e_ord, e, t in zip(event_ords, content_e, content_t):
            self.bank[e_ord.item()]['content_e'].append(e.detach().clone())
            self.bank[e_ord.item()]['content_t'].append(t.detach().clone())
            if len(self.bank[e_ord.item()]['content_e']) > self.max_size:
                self.bank[e_ord.item()]['content_e'].popleft()
                self.bank[e_ord.item()]['content_t'].popleft()

    def get_positive(self, event_ord):
        if event_ord.dim() == 0:
            event_ord = event_ord.unsqueeze(0)
    
        positive_samples = []
        for ord_ in event_ord:
            if ord_.item() in self.bank:
                positive_samples.append({
                    'content_e': torch.stack(list(self.bank[ord_.item()]['content_e'])),
                    'content_t': torch.stack(list(self.bank[ord_.item()]['content_t']))
                })
        if positive_samples:
            return positive_samples
        return None

    def get_negative(self, exclude_ord, num=100):
        all_events = [k for k in self.bank.keys() if k not in exclude_ord]
        sampled_events = random.sample(all_events, min(num, len(all_events)))
        if not sampled_events:
            return None
    
        neg_content_e = []
        neg_content_t = []
    
        for e in sampled_events:

            content_e_list = list(self.bank[e]['content_e'])
            content_t_list = list(self.bank[e]['content_t'])

            content_e_list = [e for e in content_e_list if e.size(0) == content_e_list[0].size(0)]
            content_t_list = [t for t in content_t_list if t.size(0) == content_t_list[0].size(0)]

            if content_e_list and content_t_list:
                neg_content_e.append(torch.stack(content_e_list))
                neg_content_t.append(torch.stack(content_t_list))
    
        if neg_content_e and neg_content_t:
            return {
                'content_e': torch.cat(neg_content_e),
                'content_t': torch.cat(neg_content_t)
            }
        return None

    def compute_event_contrastive_loss(self, event_ord, content_e, content_t):
        losses = []
        batch_size = content_e.shape[0]
        for i in range(batch_size):
            ord_i = event_ord[i]
            pos = self.get_positive(ord_i)
            
            if pos is None:
                losses.append(torch.tensor(0.0, device=content_e.device))
                continue
    
            if isinstance(pos, list):
                pos = pos[0]

            ce = content_e[i]  # current sample content_e
            ct = content_t[i]  # current sample content_t
    
            ce_exp = ce.unsqueeze(0).expand(pos['content_t'].shape[0], -1)
            pos_sim_e_t = F.cosine_similarity(ce_exp, pos['content_t'], dim=-1) / self.tau
    
            ct_exp = ct.unsqueeze(0).expand(pos['content_e'].shape[0], -1)
            pos_sim_t_e = F.cosine_similarity(ct_exp, pos['content_e'], dim=-1) / self.tau
    
            pos_sum = torch.sum(torch.exp(pos_sim_e_t)) + torch.sum(torch.exp(pos_sim_t_e))
    
            # Get negative samples
            neg = self.get_negative(ord_i, num=100)
            if neg is None:
                losses.append(torch.tensor(0.0, device=content_e.device))
                continue
    
            # Negative samples similarity
            ce_exp_neg = ce.unsqueeze(0).expand(neg['content_t'].shape[0], -1)
            neg_sim_e_t = F.cosine_similarity(ce_exp_neg, neg['content_t'], dim=-1) / self.tau
    
            ct_exp_neg = ct.unsqueeze(0).expand(neg['content_e'].shape[0], -1)
            neg_sim_t_e = F.cosine_similarity(ct_exp_neg, neg['content_e'], dim=-1) / self.tau
    
            neg_sum = torch.sum(torch.exp(neg_sim_e_t)) + torch.sum(torch.exp(neg_sim_t_e))
    
            # Compute final loss
            loss_i = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
            losses.append(loss_i)
    
        return torch.mean(torch.stack(losses))
    
    def compute_batch_contrastive_loss(self, event_ords, modality_features, event_features):

        total_loss = 0.0
        modality_count = 0
        device = list(modality_features.values())[0].device

        for modality, mod_features in modality_features.items():
            if modality in event_features:
                evt_features = event_features[modality]
                modality_loss = self.compute_event_contrastive_loss(event_ords, evt_features, mod_features)
                total_loss += modality_loss
                modality_count += 1

        if modality_count == 0:
            return torch.tensor(0.0, device=device)
            
        return total_loss / modality_count
        
    def clear(self):
        self.bank = defaultdict(lambda: {'content_e': deque(), 'content_t': deque()})


# 视觉-事件语义增强模块 (对应论文中的ESA部分)
class EventAwareVisual(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareVisual, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        # 模态特定MLP(论文公式1)
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_v = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 多模态自注意力(MMA)和多头交叉注意力(MHA)
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        # MLP加平均池化(论文公式3)
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_v = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 层归一化
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_v = nn.LayerNorm(128)
        
        # 局部分类器
        self.content_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        
        # 内存库
        self.memory_bank_v = EventMemoryBank(max_size=1000)

    def forward(self, memory_bank_ready, **kwargs):
        visual_fea = kwargs['visual_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        # 1. MLP映射得到h_e和h_v (论文公式1)
        h_e = self.mlp_e(event_fea)  # 事件特征
        h_v = self.mlp_v(visual_fea)  # 视觉特征
        
        # 2. 在时间维度上拼接特征，得到h_f (论文段落中的concatenation operation)
        # 注意：这里考虑各模态特征的序列长度，需要适当处理以便拼接
        # 假设h_e: [batch_size, seq_len_e, 128], h_v: [batch_size, seq_len_v, 128]
        batch_size = h_e.shape[0]
        seq_len_e = h_e.shape[1]
        seq_len_v = h_v.shape[1]
        
        # 使用均值池化将h_e压缩为单个向量（如果h_e是序列）
        if seq_len_e > 1:
            h_e_pooled = h_e.mean(dim=1, keepdim=True)  # [batch_size, 1, 128]
            h_e_expanded = h_e_pooled.expand(-1, seq_len_v, -1)  # [batch_size, seq_len_v, 128]
            h_f = torch.cat([h_e_expanded, h_v], dim=2)  # [batch_size, seq_len_v, 256]
        else:
            h_e_expanded = h_e.expand(-1, seq_len_v, -1)  # [batch_size, seq_len_v, 128]
            h_f = torch.cat([h_e_expanded, h_v], dim=2)  # [batch_size, seq_len_v, 256]
        
        # 3. 多模态自注意力(MMA)处理h_f (论文公式2的第一部分)
        # 调整维度以匹配MMA输入
        h_f_projected = nn.Linear(h_f.shape[2], 128).to(h_f.device)(h_f)
        h_f = self.mma(h_f_projected)  # [batch_size, seq_len_v, 128]
        
        # 4. 多头交叉注意力(MHA)处理h_e, h_v和h_f (论文公式2的第二部分)
        # 由于h_e维度可能与h_v不同，需要调整
        if seq_len_e != seq_len_v:
            h_e_expand = h_e.mean(dim=1, keepdim=True).expand(-1, seq_len_v, -1)
        else:
            h_e_expand = h_e
            
        h_e_out, _, h_v_out = self.mha(query=h_e_expand, key=h_v, value=h_f)
        
        # 5. 平均池化和层归一化得到全局表示g_e和g_v (论文公式3)
        h_e_mean = h_e_out.mean(dim=1)  # [batch_size, 128]
        h_v_mean = h_v_out.mean(dim=1)  # [batch_size, 128]
        
        g_e_v = self.layernorm_e(self.mlp_g_e(h_e_mean))  # [batch_size, 128]
        g_v = self.layernorm_v(self.mlp_g_v(h_v_mean))  # [batch_size, 128]
        
        # 6. 存储到memory bank用于事件对比学习
        self.memory_bank_v.push(event_ord, g_e_v, g_v)
        
        # 7. 局部分类
        # 这里使用视觉特征g_v进行分类
        output_local_v = self.content_classifier(g_v)

        return output_local_v, g_v, g_e_v


# 文本-事件语义增强模块 (对应论文中的ESA部分)
class EventAwareText(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareText, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        # 模态特定MLP(论文公式1)
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_t = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 多模态自注意力(MMA)和多头交叉注意力(MHA)
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        # MLP加平均池化(论文公式3)
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_t = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 层归一化
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_t = nn.LayerNorm(128)
        
        # 局部分类器
        self.content_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        
        # 内存库
        self.memory_bank_t = EventMemoryBank(max_size=1000)

    def forward(self, memory_bank_ready, **kwargs):
        text_fea = kwargs['text_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        # 1. MLP映射得到h_e和h_t (论文公式1)
        h_e = self.mlp_e(event_fea)  # 事件特征
        h_t = self.mlp_t(text_fea)  # 文本特征
        
        # 2. 在时间维度上拼接特征，得到h_f (论文段落中的concatenation operation)
        batch_size = h_e.shape[0]
        seq_len_e = h_e.shape[1]
        seq_len_t = h_t.shape[1]
        
        # 使用均值池化将h_e压缩为单个向量（如果h_e是序列）
        if seq_len_e > 1:
            h_e_pooled = h_e.mean(dim=1, keepdim=True)  # [batch_size, 1, 128]
            h_e_expanded = h_e_pooled.expand(-1, seq_len_t, -1)  # [batch_size, seq_len_t, 128]
            h_f = torch.cat([h_e_expanded, h_t], dim=2)  # [batch_size, seq_len_t, 256]
        else:
            h_e_expanded = h_e.expand(-1, seq_len_t, -1)  # [batch_size, seq_len_t, 128]
            h_f = torch.cat([h_e_expanded, h_t], dim=2)  # [batch_size, seq_len_t, 256]
        
        # 3. 多模态自注意力(MMA)处理h_f (论文公式2的第一部分)
        # 调整维度以匹配MMA输入
        h_f_projected = nn.Linear(h_f.shape[2], 128).to(h_f.device)(h_f)
        h_f = self.mma(h_f_projected)  # [batch_size, seq_len_t, 128]
        
        # 4. 多头交叉注意力(MHA)处理h_e, h_t和h_f (论文公式2的第二部分)
        # 由于h_e维度可能与h_t不同，需要调整
        if seq_len_e != seq_len_t:
            h_e_expand = h_e.mean(dim=1, keepdim=True).expand(-1, seq_len_t, -1)
        else:
            h_e_expand = h_e
            
        h_e_out, _, h_t_out = self.mha(query=h_e_expand, key=h_t, value=h_f)
        
        # 5. 平均池化和层归一化得到全局表示g_e和g_t (论文公式3)
        h_e_mean = h_e_out.mean(dim=1)  # [batch_size, 128]
        h_t_mean = h_t_out.mean(dim=1)  # [batch_size, 128]
        
        g_e_t = self.layernorm_e(self.mlp_g_e(h_e_mean))  # [batch_size, 128]
        g_t = self.layernorm_t(self.mlp_g_t(h_t_mean))  # [batch_size, 128]
        
        # 6. 存储到memory bank用于事件对比学习
        self.memory_bank_t.push(event_ord, g_e_t, g_t)
        
        # 7. 局部分类
        # 这里使用文本特征g_t进行分类
        output_local_t = self.content_classifier(g_t)

        return output_local_t, g_t, g_e_t


# 音频-事件语义增强模块 (对应论文中的ESA部分)
class EventAwareAudio(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareAudio, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        # 模态特定MLP(论文公式1)
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_a = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 多模态自注意力(MMA)和多头交叉注意力(MHA)
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        # MLP加平均池化(论文公式3)
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_a = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        # 层归一化
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_a = nn.LayerNorm(128)
        
        # 局部分类器
        self.content_classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        
        # 内存库
        self.memory_bank_a = EventMemoryBank(max_size=1000)

    def forward(self, memory_bank_ready, **kwargs):
        audio_fea = kwargs['audio_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        # 1. MLP映射得到h_e和h_a (论文公式1)
        h_e = self.mlp_e(event_fea)  # 事件特征
        h_a = self.mlp_a(audio_fea).unsqueeze(1)  # 音频特征 [batch_size, 1, 128]
        
        # 2. 在时间维度上拼接特征，得到h_f (论文段落中的concatenation operation)
        batch_size = h_e.shape[0]
        seq_len_e = h_e.shape[1]
        seq_len_a = h_a.shape[1]  # 通常是1
        
        # 由于音频特征往往是一个向量，我们需要调整事件特征以便拼接
        if seq_len_e > 1:
            h_e_pooled = h_e.mean(dim=1, keepdim=True)  # [batch_size, 1, 128]
            h_f = torch.cat([h_e_pooled, h_a], dim=2)  # [batch_size, 1, 256]
        else:
            h_f = torch.cat([h_e, h_a], dim=2)  # [batch_size, 1, 256]
        
        # 3. 多模态自注意力(MMA)处理h_f (论文公式2的第一部分)
        # 调整维度以匹配MMA输入
        h_f_projected = nn.Linear(h_f.shape[2], 128).to(h_f.device)(h_f)
        h_f = self.mma(h_f_projected)  # [batch_size, 1, 128]
        
        # 4. 多头交叉注意力(MHA)处理h_e, h_a和h_f (论文公式2的第二部分)
        # 由于h_e维度可能与h_a不同，需要调整
        if seq_len_e > 1:
            h_e_expand = h_e_pooled  # 使用池化后的事件特征
        else:
            h_e_expand = h_e
            
        h_e_out, _, h_a_out = self.mha(query=h_e_expand, key=h_a, value=h_f)
        
        # 5. 平均池化和层归一化得到全局表示g_e和g_a (论文公式3)
        h_e_mean = h_e_out.mean(dim=1)  # [batch_size, 128]
        h_a_mean = h_a_out.squeeze(1) if h_a_out.shape[1] == 1 else h_a_out.mean(dim=1)  # [batch_size, 128]
        
        g_e_a = self.layernorm_e(self.mlp_g_e(h_e_mean))  # [batch_size, 128]
        g_a = self.layernorm_a(self.mlp_g_a(h_a_mean))  # [batch_size, 128]
        
        # 6. 存储到memory bank用于事件对比学习
        self.memory_bank_a.push(event_ord, g_e_a, g_a)
        
        # 7. 局部分类
        # 这里使用音频特征g_a进行分类
        output_local_a = self.content_classifier(g_a)

        return output_local_a, g_a, g_e_a


# 全局尺度检测模块 (对应论文中的Global-scale Detection)
class GlobalScaleModule(torch.nn.Module):
    def __init__(self, dataset):
        super(GlobalScaleModule, self).__init__()
        # 全局分类器
        self.global_classifier = nn.Sequential(nn.Linear(128 * 4, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 2))

    def forward(self, g_v, g_t, g_a, g_e):
        # 将所有模态的全局表示和事件表示拼接，形成全局表示
        global_fea = torch.cat((g_v, g_t, g_a, g_e), dim=1)
        
        # 使用全局分类器进行分类
        output_global = self.global_classifier(global_fea)

        return output_global


# EAM模型整体结构
class EAM_Model(torch.nn.Module):
    def __init__(self, dataset):
        super(EAM_Model, self).__init__()
        # 事件感知语义增强模块 (ESA)
        self.visual_branch = EventAwareVisual(dataset=dataset)
        self.text_branch = EventAwareText(dataset=dataset)
        self.audio_branch = EventAwareAudio(dataset=dataset)

        # 全局尺度检测模块
        self.global_branch = GlobalScaleModule(dataset=dataset)
        
        # 局部尺度分类器
        self.local_classifier = nn.Linear(3, 2)
        
        # 统一的事件对比学习内存库
        self.event_memory_bank = EventMemoryBank(max_size=1000)

    def forward(self, memory_bank_ready, **kwargs):
        # ESA: 事件感知语义增强处理各模态
        output_local_v, g_v, g_e_v = self.visual_branch(memory_bank_ready, **kwargs)
        output_local_t, g_t, g_e_t = self.text_branch(memory_bank_ready, **kwargs)
        output_local_a, g_a, g_e_a = self.audio_branch(memory_bank_ready, **kwargs)

        # 局部尺度检测: 融合三个模态的局部分类结果
        local_logits = torch.stack([
            output_local_v[:, 1], 
            output_local_t[:, 1], 
            output_local_a[:, 1]
        ], dim=1)
        output_local = self.local_classifier(local_logits)

        # 全局尺度检测: 使用所有模态和事件的全局表示
        g_e = (g_e_v + g_e_t + g_e_a) / 3  # 事件表示平均
        output_global = self.global_branch(g_v, g_t, g_a, g_e)

        # 最终输出是局部和全局的平均
        output = 0.5 * (output_local + output_global)

        # 事件对比学习损失 - 每个batch计算一次
        event_cl_loss = 0.0
        if memory_bank_ready:
            # 收集所有模态特征和事件特征
            modality_features = {
                'visual': g_v,
                'text': g_t,
                'audio': g_a
            }
            event_features = {
                'visual': g_e_v,
                'text': g_e_t,
                'audio': g_e_a
            }
            
            # 为整个batch计算一次事件对比损失
            event_cl_loss = self.event_memory_bank.compute_batch_contrastive_loss(
                kwargs['event_ord'], 
                modality_features, 
                event_features
            )

        return output, output_global, output_local, event_cl_loss
