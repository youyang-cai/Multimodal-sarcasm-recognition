import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import torch.optim as optim
import warnings
import argparse
import time
import random
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report

import torch.nn as nn
import pickle

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载提取的特征数据
with open('processed_data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

with open('processed_data/text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)

with open('processed_data/visual_features.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# 从MUSTARD.xlsx中加载标签数据
data = pd.read_excel('MUSTARD/MUSTARD.xlsx')
labels_dict = {row['KEY']: row['SARCASM'] for _, row in data.iterrows()}


# 打印一些信息来确认加载成功
print("Loaded audio features:", len(audio_features))
print("Loaded text features:", len(text_features))
print("Loaded visual features:", len(visual_features))

class FusionDataset(Dataset):
    def __init__(self, text_features, audio_features, visual_features, labels_dict, device, max_audio_len=1000):
        self.text_features = text_features
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.labels_dict = labels_dict
        self.device = device
        self.max_audio_len = max_audio_len
        self.keys = [key for key in text_features.keys() if key in audio_features and key in visual_features and key in labels_dict]
        print(f"Number of valid keys: {len(self.keys)}")  # 打印有效键的数量

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        text_feature = self.text_features[key]['text'].squeeze().to(self.device)
        audio_feature = torch.tensor(self.audio_features[key]['librosa'], dtype=torch.float32).flatten().to(self.device)
        visual_feature = self.visual_features[key]['visual'].squeeze().to(self.device)

        # 填充或截断音频特征
        if audio_feature.size(0) > self.max_audio_len:
            audio_feature = audio_feature[:self.max_audio_len]
        else:
            padding = torch.zeros(self.max_audio_len - audio_feature.size(0), device=self.device)
            audio_feature = torch.cat((audio_feature, padding))

        # 从标签字典中获取标签
        label = self.labels_dict[key]
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        # 拼接特征
        feature = torch.cat((text_feature, audio_feature, visual_feature))

        return feature, label

# 创建数据集
dataset = FusionDataset(text_features, audio_features, visual_features, labels_dict, device)

# 划分数据集
def get_train_valid_test_sampler(dataset_size, valid_ratio=0.1, test_ratio=0.1):
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split1 = int(np.floor(valid_ratio * dataset_size))
    split2 = int(np.floor(test_ratio * dataset_size))
    train_indices, valid_indices, test_indices = indices[split2:], indices[:split1], indices[split1:split2]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices), SubsetRandomSampler(test_indices)

train_sampler, valid_sampler, test_sampler = get_train_valid_test_sampler(len(dataset))

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# 打印一些信息来确认数据集和数据加载器
print("Dataset size:", len(dataset))

class EarlyFusionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EarlyFusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# 参数设置
input_size = 768 + 2048 + 1000  # text + visual + audio
hidden_size = 512
num_classes = 2  # 假定为二分类问题

model = EarlyFusionMLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 打印模型结构
print(model)

def train_model(model, criterion, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    true_labels = []
    predicted_labels = []

    for features, labels in data_loader:
        # 确保所有张量都在同一个设备上
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall, f1 = evaluate_performance(true_labels, predicted_labels)
    return avg_loss, precision, recall, f1

def evaluate_model(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            # 确保所有张量都在同一个设备上
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall, f1 = evaluate_performance(true_labels, predicted_labels)
    return avg_loss, precision, recall, f1, true_labels, predicted_labels

def evaluate_performance(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return precision, recall, f1

# 开始训练
num_epochs = 200

for epoch in range(num_epochs):
    train_loss, train_precision, train_recall, train_f1 = train_model(model, criterion, optimizer, train_loader, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.6f}, Precision: {train_precision:.6f}, Recall: {train_recall:.6f}, F1: {train_f1:.6f}')

# 评估模型
valid_loss, valid_precision, valid_recall, valid_f1, valid_true_labels, valid_predicted_labels = evaluate_model(model, criterion, valid_loader, device)
print(f'Validation, Loss: {valid_loss:.6f}, Precision: {valid_precision:.6f}, Recall: {valid_recall:.6f}, F1: {valid_f1:.6f}')
print('Validation Classification Report:')
print(classification_report(valid_true_labels, valid_predicted_labels))

test_loss, test_precision, test_recall, test_f1, test_true_labels, test_predicted_labels = evaluate_model(model, criterion, test_loader, device)
print(f'Test, Loss: {test_loss:.6f}, Precision: {test_precision:.6f}, Recall: {test_recall:.6f}, F1: {test_f1:.6f}')
print('Test Classification Report:')
print(classification_report(test_true_labels, test_predicted_labels))
print('Confusion Matrix:')
print(confusion_matrix(test_true_labels, test_predicted_labels))
