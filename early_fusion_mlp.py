import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

# Load extracted features
with open('processed_data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)
with open('processed_data/text_features.pkl', 'rb') as f:
    text_features = pickle.load(f)
with open('processed_data/visual_features.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# Print a sample to see the data structure
print("Sample text feature keys:", list(text_features.values())[0].keys())
print("Sample audio feature keys:", list(audio_features.values())[0].keys())
print("Sample visual feature keys:", list(visual_features.values())[0].keys())

# Load labels
data = pd.read_excel('MUSTARD/MUSTARD.xlsx')
labels_dict = {row['KEY']: row['SARCASM'] for _, row in data.iterrows()}

class FusionDataset(Dataset):
    def __init__(self, text_features, audio_features, visual_features, labels_dict, device, max_audio_len=1000):
        self.text_features = text_features
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.labels_dict = labels_dict
        self.device = device
        self.max_audio_len = max_audio_len
        self.keys = [key for key in text_features.keys() if key in audio_features and key in visual_features and key in labels_dict]
        print(f"Number of valid keys: {len(self.keys)}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        text_feature = self.text_features[key]['text'].squeeze().to(self.device)
        audio_feature = torch.tensor(self.audio_features[key]['librosa'], dtype=torch.float32).flatten().to(self.device)
        visual_feature = self.visual_features[key]['visual'].squeeze().to(self.device)
        if audio_feature.size(0) > self.max_audio_len:
            audio_feature = audio_feature[:self.max_audio_len]
        else:
            padding = torch.zeros(self.max_audio_len - audio_feature.size(0), device=self.device)
            audio_feature = torch.cat((audio_feature, padding))
        label = self.labels_dict[key]
        label = torch.tensor(label, dtype=torch.long).to(self.device)
        feature = torch.cat((text_feature, audio_feature, visual_feature))
        return feature, label

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct_predictions / len(train_loader.dataset)
    return avg_loss, accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = correct_predictions / len(val_loader.dataset)
    return avg_loss, accuracy


def perform_cross_validation(dataset, input_size, n_splits=5, hidden_size=512, output_size=2):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        train_sampler = SubsetRandomSampler(train_index)
        test_sampler = SubsetRandomSampler(test_index)
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)

        model = MLP(input_size, hidden_size, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"Training fold {fold+1}/{n_splits}")
        for epoch in range(10):  # Assume 10 epochs for simplicity
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_model(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Revised prediction and reporting logic using the test_loader properly
        actual_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                actual_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        report = classification_report(actual_labels, predicted_labels, output_dict=True)
        results.append(report)
        print(classification_report(actual_labels, predicted_labels))

    # Calculate average metrics across folds
    avg_precision = np.mean([result['weighted avg']['precision'] for result in results])
    avg_recall = np.mean([result['weighted avg']['recall'] for result in results])
    avg_f1_score = np.mean([result['weighted avg']['f1-score'] for result in results])

    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1 Score: {avg_f1_score:.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FusionDataset(text_features, audio_features, visual_features, labels_dict, device)
    sample_feature, _ = dataset[0]
    input_size = sample_feature.shape[0]
    print(f"Input size of the model should be: {input_size}")
    perform_cross_validation(dataset, input_size=input_size)

