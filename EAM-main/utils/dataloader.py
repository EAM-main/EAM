import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

class EAM_Dataset(Dataset):
    def __init__(self, vid_path, dataset):
        self.dataset = dataset
        if dataset == 'fakesv':
            self.data_all = pd.read_json('./fea/fakesv/metainfo.json', orient='records', dtype=False, lines=True)
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            with open('./fea/fakesv/fakesv.json', 'r') as f:
                event_data = [json.loads(line.strip()) for line in f.readlines()]
            event_df = pd.DataFrame(event_data)
            event_df = event_df[event_df.video_id.isin(self.vid)]
            self.data = pd.merge(self.data, event_df[['video_id', 'event_ord']], on='video_id', how='left')
            self.data['event_ord'] = self.data['event_ord'].fillna(-1).astype(int)

            self.text_semantic_path = './fea/fakesv/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_path, 'rb') as f:
                self.text_fea = torch.load(f)

            self.audio_path = './fea/fakesv/preprocess_audio'
            self.visual_path = './fea/fakesv/preprocess_visual'

            self.event_path = './fea/fakesv/sem_event_fea.pkl'
            with open(self.event_path, 'rb') as f:
                self.event_fea = torch.load(f)

        elif dataset == 'fakett':
            self.data_all = pd.read_json('./fea/fakett/metainfo.json', orient='records', lines=True,
                                         dtype={'video_id': str})
            self.vid = []
            with open(vid_path, "r") as fr:
                for line in fr.readlines():
                    self.vid.append(line.strip())
            self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
            self.data.reset_index(inplace=True)

            with open('./fea/fakett/fakett.json', 'r') as f:
                event_data = [json.loads(line.strip()) for line in f.readlines()]
            event_df = pd.DataFrame(event_data)
            event_df = event_df[event_df.video_id.isin(self.vid)]
            self.data = pd.merge(self.data, event_df[['video_id', 'event_ord']], on='video_id', how='left')
            self.data['event_ord'] = self.data['event_ord'].fillna(-1).astype(int)

            self.text_semantic_path = './fea/fakett/preprocess_text/sem_text_fea.pkl'
            with open(self.text_semantic_path, 'rb') as f:
                self.text_fea = torch.load(f)

            self.event_path = './fea/fakett/sem_event_fea.pkl'
            with open(self.event_path, 'rb') as f:
                self.event_fea = torch.load(f)

            self.audio_path = './fea/fakett/preprocess_audio'
            self.visual_path = './fea/fakett/preprocess_visual'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        vid = item['video_id']
        label = 1 if item['annotation'] == 'fake' else 0
        event_ord = item['event_ord']

        label = torch.tensor(label)
        event_ord = torch.tensor(event_ord)

        text_fea = self.text_fea['last_hidden_state'][vid]
        event_fea = self.event_fea[vid]

        v_path = os.path.join(self.visual_path, vid + '.pkl')
        visual_fea = torch.tensor(torch.load(open(v_path, 'rb')))

        a_path = os.path.join(self.audio_path, vid + '.pkl')
        audio_fea = torch.load(open(a_path, 'rb'))

        return {
            'vid': vid,
            'label': label,
            'event_ord': event_ord,
            'text_fea': text_fea,
            'visual_fea': visual_fea,
            'audio_fea': audio_fea,
            'event_fea': event_fea
        }


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def collate_fn_EAM(batch):
    num_visual_frames = 83

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])
    event_ord = torch.stack([item['event_ord'] for item in batch])

    text_fea = [item['text_fea'] for item in batch]
    visual_fea = [item['visual_fea'] for item in batch]
    audio_fea = [item['audio_fea'] for item in batch]
    event_fea = torch.stack([item['event_fea'] for item in batch])

    visual_fea, _ = pad_frame_sequence(num_visual_frames, visual_fea)
    audio_fea = torch.cat(audio_fea, dim=0)

    text_fea = [x if x.shape[0] == 512 else torch.cat((x, torch.zeros([512 - x.shape[0], x.shape[1]], dtype=torch.float)),
                                         dim=0) for x in text_fea]
    text_fea = torch.stack(text_fea)

    return {
        'vid': vid,
        'label': label,
        'text_fea': text_fea,
        'visual_fea': visual_fea,
        'audio_fea': audio_fea,
        'event_fea': event_fea,
        'event_ord': event_ord
    }
