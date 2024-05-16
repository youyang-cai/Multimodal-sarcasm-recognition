import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle

class MUSTARDDataset(Dataset):
    def __init__(self, path, train=True):
        with open(path, 'rb') as file:
            allfeature = pickle.load(file, encoding='latin1')
        
        # 根据您的描述，假设allfeature中的数据索引如下
        self.videoIDs = allfeature[0]
        self.videoSpeakers = allfeature[1]  # 假设是视频发言人特征
        self.sarcasmsLabels = allfeature[2]  # 假设是讽刺标签
        self.sentimentImplicit = allfeature[3]
        self.sentimentExplicit = allfeature[4]
        self.videoText = allfeature[7]
        self.videoAudio = allfeature[8]
        self.videoVisual = allfeature[9]
        self.videoSentence = allfeature[10]
        self.trainVid = sorted(allfeature[11])
        self.testVid = sorted(allfeature[12])
        self.keys = self.trainVid if train else self.testVid

    def __getitem__(self, index):
        vid = self.keys[index]
        umask = [0] * (len(self.sarcasmsLabels[vid]) - 1) + [1]

        text = torch.tensor(self.videoText[vid][0], dtype=torch.float)
        audio = torch.tensor(self.videoAudio[vid][0], dtype=torch.float)
        visual = torch.tensor(self.videoVisual[vid][0], dtype=torch.float)
        
        if visual.shape[0] != 2048:
            raise ValueError(f"Feature dimension mismatch for video {vid}: expected 2048, got {visual.shape[0]}")

        speakers = torch.tensor(self.videoSpeakers[vid], dtype=torch.float)
        sarcasm_labels = torch.tensor(self.sarcasmsLabels[vid], dtype=torch.long)
        sentiment_imp = torch.tensor(self.sentimentImplicit[vid], dtype=torch.long)
        sentiment_exp = torch.tensor(self.sentimentExplicit[vid], dtype=torch.long)
        
        return {
            'text': text,
            'audio': audio,
            'visual': visual,
            'speakers': speakers,
            'umask': torch.tensor(umask, dtype=torch.float),
            'sarcasm_labels': sarcasm_labels,
            'sentiment_imp': sentiment_imp,
            'sentiment_exp': sentiment_exp,
            'vid': vid
        }

    def __len__(self):
        return len(self.keys)

    def collate_fn(self, batch):
        # 此处实现按需批处理数据的逻辑
        batch_text = pad_sequence([item['text'] for item in batch], batch_first=True)
        batch_audio = pad_sequence([item['audio'] for item in batch], batch_first=True)
        batch_visual = pad_sequence([item['visual'] for item in batch], batch_first=True)
        batch_speakers = pad_sequence([item['speakers'] for item in batch], batch_first=True)
        batch_umask = torch.stack([item['umask'] for item in batch])
        batch_sarcasm_labels = torch.stack([item['sarcasm_labels'] for item in batch])
        batch_sentiment_imp = torch.stack([item['sentiment_imp'] for item in batch])
        batch_sentiment_exp = torch.stack([item['sentiment_exp'] for item in batch])
        batch_vid = [item['vid'] for item in batch]

        return {
            'text': batch_text,
            'audio': batch_audio,
            'visual': batch_visual,
            'speakers': batch_speakers,
            'umask': batch_umask,
            'sarcasm_labels': batch_sarcasm_labels,
            'sentiment_imp': batch_sentiment_imp,
            'sentiment_exp': batch_sentiment_exp,
            'vid': batch_vid
        }
        
    # def collate_fn(batch):
    #     batch_data = {'text': [], 'visual': [], 'audio': [], 'umask': [], 'sarcasm_labels': [], 'sentiment_imp': [], 'sentiment_exp': []}
    #     for item in batch:
    #         batch_data['text'].append(item['text'])
    #         batch_data['visual'].append(item['visual'])
    #         batch_data['audio'].append(item['audio'])
    #         batch_data['umask'].append(item['umask'])
    #         batch_data['sarcasm_labels'].append(item['sarcasm_labels'])
    #         batch_data['sentiment_imp'].append(item['sentiment_imp'])
    #         batch_data['sentiment_exp'].append(item['sentiment_exp'])
    #     # 转换为适合的tensor
    #     for key in batch_data:
    #         if key in ['text', 'visual', 'audio']:  # 假设这些需要pad
    #             batch_data[key] = pad_sequence(batch_data[key], batch_first=True)
    #         else:
    #             batch_data[key] = torch.stack(batch_data[key])
    #     return batch_data

