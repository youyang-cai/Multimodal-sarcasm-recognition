import pickle
import torch

# 加载提取的特征数据
with open('processed_data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

with open('processed_data/text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)

with open('processed_data/visual_features.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# 打印一些信息来确认加载成功
print("Loaded audio features:", len(audio_features))
print("Loaded text features:", len(text_features))
print("Loaded visual features:", len(visual_features))

from torch.utils.data import Dataset, DataLoader

class FusionDataset(Dataset):
    def __init__(self, text_features, audio_features, visual_features):
        self.text_features = text_features
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.keys = list(text_features.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        text_feature = self.text_features[key]['text'].squeeze()
        audio_feature = torch.tensor(self.audio_features[key]['librosa'], dtype=torch.float32).flatten()
        visual_feature = self.visual_features[key]['visual'].squeeze()

        label = int(key.split('_')[-1])  # Assuming label is part of the key, e.g., 'video_0', 'video_1'
        label = torch.tensor(label, dtype=torch.long)

        # 拼接特征
        feature = torch.cat((text_feature, audio_feature, visual_feature))

        return feature, label

# 创建数据集和数据加载器
dataset = FusionDataset(text_features, audio_features, visual_features)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 打印一些信息来确认数据集和数据加载器
print("Dataset size:", len(dataset))
