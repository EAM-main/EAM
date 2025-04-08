import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MultiModalAttention import MultiModalAttention
from model.MultiHeadCrossAttention import MultiHeadCrossAttention
from collections import defaultdict, deque
import random

class EventMemoryBank:
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.bank = defaultdict(lambda: {'g_e': deque(), 'g_x': deque()})
        self.tau = 0.8

    def push(self, event_ords, g_e, g_x):
        for e_ord, e, x in zip(event_ords, g_e, g_x):
            self.bank[e_ord.item()]['g_e'].append(e.detach().clone())
            self.bank[e_ord.item()]['g_x'].append(x.detach().clone())
            if len(self.bank[e_ord.item()]['g_e']) > self.max_size:
                self.bank[e_ord.item()]['g_e'].popleft()
                self.bank[e_ord.item()]['g_x'].popleft()

    def get_positive(self, event_ord):
        if event_ord.dim() == 0:
            event_ord = event_ord.unsqueeze(0)
    
        positive_samples = []
        for ord_ in event_ord:
            if ord_.item() in self.bank:
                positive_samples.append({
                    'g_e': torch.stack(list(self.bank[ord_.item()]['g_e'])),
                    'g_x': torch.stack(list(self.bank[ord_.item()]['g_x']))
                })
        if positive_samples:
            return positive_samples
        return None

    def get_negative(self, exclude_ord, num=100):
        all_events = [k for k in self.bank.keys() if k not in exclude_ord]
        sampled_events = random.sample(all_events, min(num, len(all_events)))
        if not sampled_events:
            return None
    
        neg_g_e = []
        neg_g_x = []
    
        for e in sampled_events:
            g_e_list = list(self.bank[e]['g_e'])
            g_x_list = list(self.bank[e]['g_x'])

            g_e_list = [e for e in g_e_list if e.size(0) == g_e_list[0].size(0)]
            g_x_list = [x for x in g_x_list if x.size(0) == g_x_list[0].size(0)]

            if g_e_list and g_x_list:
                neg_g_e.append(torch.stack(g_e_list))
                neg_g_x.append(torch.stack(g_x_list))
    
        if neg_g_e and neg_g_x:
            return {
                'g_e': torch.cat(neg_g_e),
                'g_x': torch.cat(neg_g_x)
            }
        return None

    def compute_event_contrastive_loss(self, event_ord, g_e, g_x):
        losses = []
        batch_size = g_e.shape[0]
        for i in range(batch_size):
            ord_i = event_ord[i]
            pos = self.get_positive(ord_i)
            
            if pos is None:
                losses.append(torch.tensor(0.0, device=g_e.device))
                continue
    
            if isinstance(pos, list):
                pos = pos[0]

            ce = g_e[i]
            cx = g_x[i]
    
            ce_exp = ce.unsqueeze(0).expand(pos['g_x'].shape[0], -1)
            pos_sim_e_x = F.cosine_similarity(ce_exp, pos['g_x'], dim=-1) / self.tau
    
            cx_exp = cx.unsqueeze(0).expand(pos['g_e'].shape[0], -1)
            pos_sim_x_e = F.cosine_similarity(cx_exp, pos['g_e'], dim=-1) / self.tau
    
            pos_sum = torch.sum(torch.exp(pos_sim_e_x)) + torch.sum(torch.exp(pos_sim_x_e))
    
            neg = self.get_negative(ord_i, num=100)
            if neg is None:
                losses.append(torch.tensor(0.0, device=g_e.device))
                continue
    
            ce_exp_neg = ce.unsqueeze(0).expand(neg['g_x'].shape[0], -1)
            neg_sim_e_x = F.cosine_similarity(ce_exp_neg, neg['g_x'], dim=-1) / self.tau
    
            cx_exp_neg = cx.unsqueeze(0).expand(neg['g_e'].shape[0], -1)
            neg_sim_x_e = F.cosine_similarity(cx_exp_neg, neg['g_e'], dim=-1) / self.tau
    
            neg_sum = torch.sum(torch.exp(neg_sim_e_x)) + torch.sum(torch.exp(neg_sim_x_e))
    
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
        self.bank = defaultdict(lambda: {'g_e': deque(), 'g_x': deque()})

class EventAwareVisual(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareVisual, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_v = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_v = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_v = nn.LayerNorm(128)
        
        self.memory_bank_v = EventMemoryBank(max_size=1000)

    def forward(self, **kwargs):
        visual_fea = kwargs['visual_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        h_e = self.mlp_e(event_fea)
        h_v = self.mlp_v(visual_fea)
        
        h_f = torch.cat([h_e, h_v], dim=1)
        h_= self.mma(h_f)
        
        h_e_out, _, h_v_out = self.mha(query=h_e, key=h_, value=h_v)
        
        h_e_mean = h_e_out.mean(dim=1)
        h_v_mean = h_v_out.mean(dim=1)
        
        g_e_v = self.layernorm_e(self.mlp_g_e(h_e_mean))
        g_v = self.layernorm_v(self.mlp_g_v(h_v_mean))
        
        self.memory_bank_v.push(event_ord, g_e_v, g_v)

        return g_v, g_e_v

class EventAwareText(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareText, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_t = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_t = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_t = nn.LayerNorm(128)
        
        self.memory_bank_t = EventMemoryBank(max_size=1000)

    def forward(self, **kwargs):
        text_fea = kwargs['text_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        h_e = self.mlp_e(event_fea)
        h_t = self.mlp_t(text_fea)
        
        h_f = torch.cat([h_e, h_t], dim=1)
        h_ = self.mma(h_f)
        
        h_e_out, _, h_t_out = self.mha(query=h_e, key=h_, value=h_t)
        
        h_e_mean = h_e_out.mean(dim=1)
        h_t_mean = h_t_out.mean(dim=1)
        
        g_e_t = self.layernorm_e(self.mlp_g_e(h_e_mean))
        g_t = self.layernorm_t(self.mlp_g_t(h_t_mean))
        
        self.memory_bank_t.push(event_ord, g_e_t, g_t)

        return g_t, g_e_t

class EventAwareAudio(torch.nn.Module):
    def __init__(self, dataset):
        super(EventAwareAudio, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
            
        self.mlp_e = nn.Sequential(nn.Linear(self.encoded_text_semantic_fea_dim, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_a = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.mma = MultiModalAttention(d_model=128, n_heads=4, dropout=0.1)
        self.mha = MultiHeadCrossAttention(d_model=128, n_heads=4, dropout=0.1)
        
        self.mlp_g_e = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.mlp_g_a = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        
        self.layernorm_e = nn.LayerNorm(128)
        self.layernorm_a = nn.LayerNorm(128)
        
        self.memory_bank_a = EventMemoryBank(max_size=1000)

    def forward(self, **kwargs):
        audio_fea = kwargs['audio_fea']
        event_fea = kwargs['event_fea']
        event_ord = kwargs['event_ord']

        h_e = self.mlp_e(event_fea)
        h_a = self.mlp_a(audio_fea).unsqueeze(1) if audio_fea.dim() == 2 else self.mlp_a(audio_fea)
        
        h_f = torch.cat([h_e, h_a], dim=1)
        h_ = self.mma(h_f)
        
        h_e_out, _, h_a_out = self.mha(query=h_e, key=h_, value=h_a)
        
        h_e_mean = h_e_out.mean(dim=1)
        h_a_mean = h_a_out.squeeze(1) if h_a_out.shape[1] == 1 else h_a_out.mean(dim=1)
        
        g_e_a = self.layernorm_e(self.mlp_g_e(h_e_mean))
        g_a = self.layernorm_a(self.mlp_g_a(h_a_mean))
        
        self.memory_bank_a.push(event_ord, g_e_a, g_a)

        return g_a, g_e_a

class MultiScaleDetectionModule(torch.nn.Module):
    def __init__(self, d_model=128):
        super(MultiScaleDetectionModule, self).__init__()
        self.mha_v = MultiHeadCrossAttention(d_model=d_model, n_heads=4, dropout=0.1)
        self.mha_t = MultiHeadCrossAttention(d_model=d_model, n_heads=4, dropout=0.1)
        self.mha_a = MultiHeadCrossAttention(d_model=d_model, n_heads=4, dropout=0.1)
        
        self.local_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)
        )
        
        self.global_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 2)
        )

    def forward(self, g_v, g_e_v, g_t, g_e_t, g_a, g_e_a):
        _, _, l_v = self.mha_v(g_v.unsqueeze(1), g_e_v.unsqueeze(1), g_e_v.unsqueeze(1))
        _, _, l_t = self.mha_t(g_t.unsqueeze(1), g_e_t.unsqueeze(1), g_e_t.unsqueeze(1))
        _, _, l_a = self.mha_a(g_a.unsqueeze(1), g_e_a.unsqueeze(1), g_e_a.unsqueeze(1))
        
        l_v = l_v.squeeze(1)
        l_t = l_t.squeeze(1)
        l_a = l_a.squeeze(1)
        x_l = l_v + l_t + l_a
        
        y_l = self.local_classifier(x_l)

        g_e = (g_e_v + g_e_t + g_e_a) / 3
        g_global = g_v + g_t + g_a + g_e
        y_g = self.global_classifier(g_global)

        output = 0.5 * (y_l + y_g)

        return output, y_g, y_l

class EAM_Model(torch.nn.Module):
    def __init__(self, dataset):
        super(EAM_Model, self).__init__()
        self.visual_branch = EventAwareVisual(dataset=dataset)
        self.text_branch = EventAwareText(dataset=dataset)
        self.audio_branch = EventAwareAudio(dataset=dataset)
        self.multi_scale_detection = MultiScaleDetectionModule(d_model=128)
        self.event_memory_bank = EventMemoryBank(max_size=1000)

    def forward(self, **kwargs):
        g_v, g_e_v = self.visual_branch(**kwargs)
        g_t, g_e_t = self.text_branch(**kwargs)
        g_a, g_e_a = self.audio_branch(**kwargs)

        output, output_global, output_local = self.multi_scale_detection(g_v, g_e_v, g_t, g_e_t, g_a, g_e_a)

        modality_features = {'visual': g_v, 'text': g_t,'audio': g_a}
        event_features = {'visual': g_e_v,'text': g_e_t,'audio': g_e_a}
        
        event_cl_loss = self.event_memory_bank.compute_batch_contrastive_loss(
            kwargs['event_ord'], 
            modality_features, 
            event_features
        )

        return output, output_global, output_local, event_cl_loss
