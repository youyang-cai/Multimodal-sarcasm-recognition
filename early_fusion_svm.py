import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Loading extracted features
with open('processed_data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

with open('processed_data/text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)

with open('processed_data/visual_features.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# Load labels
data = pd.read_excel('MUSTARD/MUSTARD.xlsx')
labels_dict = {row['KEY']: row['SARCASM'] for _, row in data.iterrows()}

class FusionDataset(Dataset):
    def __init__(self, text_features, audio_features, visual_features, labels_dict):
        self.text_features = text_features
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.labels_dict = labels_dict
        self.keys = [key for key in text_features.keys() if key in audio_features and key in visual_features and key in labels_dict]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        
        # 文本特征处理
        if isinstance(self.text_features[key]['text'], torch.Tensor):
            text_feature = self.text_features[key]['text'].squeeze()
            if text_feature.requires_grad:
                text_feature = text_feature.detach()
            if text_feature.is_cu da:
                text_feature = text_feature.cpu()
            text_feature = text_feature.numpy()
        else:
            text_feature = self.text_features[key]['text'].squeeze()

        # 音频特征处理，确保长度一致
        target_audio_length = 1024  # 假设你想要的固定长度
        if isinstance(self.audio_features[key]['librosa'], torch.Tensor):
            audio_feature = self.audio_features[key]['librosa'].flatten()
            if audio_feature.requires_grad:
                audio_feature = audio_feature.detach()
            if audio_feature.is_cuda:
                audio_feature = audio_feature.cpu()
            audio_feature = audio_feature.numpy()
        else:
            audio_feature = self.audio_features[key]['librosa'].flatten()
        if len(audio_feature) > target_audio_length:
            audio_feature = audio_feature[:target_audio_length]
        elif len(audio_feature) < target_audio_length:
            audio_feature = np.pad(audio_feature, (0, target_audio_length - len(audio_feature)), 'constant')

        # 视觉特征处理
        if isinstance(self.visual_features[key]['visual'], torch.Tensor):
            visual_feature = self.visual_features[key]['visual'].squeeze()
            if visual_feature.requires_grad:
                visual_feature = visual_feature.detach()
            if visual_feature.is_cuda:
                visual_feature = visual_feature.cpu()
            visual_feature = visual_feature.numpy()
        else:
            visual_feature = self.visual_features[key]['visual'].squeeze()

        feature = np.concatenate((text_feature, audio_feature, visual_feature))
        label = self.labels_dict[key]
        return feature, label





# Create Dataset
dataset = FusionDataset(text_features, audio_features, visual_features, labels_dict)

# Cross Validation and SVM Classification
def perform_cross_validation(dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for train_index, test_index in kf.split(dataset):
        train_features = [dataset[i][0] for i in train_index]
        train_labels = [dataset[i][1] for i in train_index]
        test_features = [dataset[i][0] for i in test_index]
        test_labels = [dataset[i][1] for i in test_index]

        # Normalize features and train SVM
        clf = make_pipeline(StandardScaler(), svm.SVC(gamma='scale', kernel='rbf'))
        clf.fit(train_features, train_labels)

        # Predict and evaluate
        predicted_labels = clf.predict(test_features)
        report = classification_report(test_labels, predicted_labels, output_dict=True)
        results.append(report)
        print(classification_report(test_labels, predicted_labels))
        print(confusion_matrix(test_labels, predicted_labels))

    # Average results across folds
    avg_precision = np.mean([result['weighted avg']['precision'] for result in results])
    avg_recall = np.mean([result['weighted avg']['recall'] for result in results])
    avg_f1_score = np.mean([result['weighted avg']['f1-score'] for result in results])

    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1 Score: {avg_f1_score:.3f}")

if __name__ == "__main__":
    perform_cross_validation(dataset)
