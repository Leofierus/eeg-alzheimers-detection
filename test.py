import torch
import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from eeg_dataset import EEGDataset
from eeg_net import EEGNet

warnings.filterwarnings('ignore', category=RuntimeWarning)

if not os.path.exists('images'):
    os.makedirs('images')

# Model params
num_chans = 19
timepoints = 7500
num_classes = 3
F1 = 152
D = 5
F2 = 760
dropout_rate = 0.5

model_file = 'eegNet.pth'
model = EEGNet(num_channels=num_chans, timepoints=timepoints, num_classes=num_classes, F1=F1, D=D,
               F2=F2, dropout_rate=dropout_rate)
model.load_state_dict(torch.load(model_file))
print("Model loaded successfully")

data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

test_data = [d for d in data_info if d['type'] == 'train']
test_dataset = EEGDataset(data_dir, test_data)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

total_a = 0
total_c = 0
total_f = 0
for entry in test_data:
    if entry['label'] == 'A':
        total_a += 1
    elif entry['label'] == 'C':
        total_c += 1
    else:
        total_f += 1

# Print test_dataloader info
print(f'Test dataset: {len(test_dataset)} samples')
print(f'Test dataloader: {len(test_dataloader)} batches')
print(f'Test dataloader batch size: {test_dataloader.batch_size}\n')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

all_labels = []
all_probs = []

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    correct_a = 0
    correct_c = 0
    correct_f = 0
    for eeg_data, labels in test_dataloader:
        eeg_data, labels = eeg_data.to(device), labels.to(device)
        outputs = model.predict(eeg_data)
        temp, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs[:, 1].cpu().numpy())

        for i in range(labels.size(0)):
            if labels[i] == 0 and predicted[i] == 0:
                correct_a += 1
            elif labels[i] == 1 and predicted[i] == 1:
                correct_c += 1
            elif labels[i] == 2 and predicted[i] == 2:
                correct_f += 1
            print(f'Predicted: {predicted[i]}, Model value: {outputs[i]}, Actual: {labels[i]}')

    print(f'\nCorrect: {correct}, Total: {total}')
    print(f'Correct A: {correct_a}, Total A: {total_a}')
    print(f'Correct C: {correct_c}, Total C: {total_c}')
    print(f'Correct F: {correct_f}, Total F: {total_f}')
    print(f'Accuracy: {100 * correct / total:.4f}%')

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print(f"Labels: {all_labels}\nProbs: {all_probs}")

"""
Train 1: (Good)
Epochs: 300
Learning rate: 0.001
Batch size: 8
F1=64, D=2, F2=128, dropout_rate=0.5
timepoints: 30000
Time taken: ~ 6-7 hours
Correct: 133, Total: 222
Correct A: 45, Total A: 82
Correct C: 88, Total C: 140
Accuracy: 59.9099%

Train 2: (Overfit to A)
Epochs: 1000
Learning rate: 0.001
Batch size: 16
F1=64, D=2, F2=128, dropout_rate=0.5
timepoints: 30000
Time taken: ~ 2 days
Correct: 82, Total: 222
Correct A: 82, Total A: 82
Correct C: 0, Total C: 140
Accuracy: 36.9369%

Train 3: (Overfit to C)
Epochs: 180
Learning rate: 0.001
Batch size: 20
F1=64, D=2, F2=128, dropout_rate=0.5
timepoints: 30000
Time taken: ~ 12 hours
Correct: 140, Total: 222
Correct A: 0, Total A: 82
Correct C: 140, Total C: 140
Accuracy: 63.0631%

Train 4: (Overfit to C)
Epochs: 10
Learning rate: 0.001
Batch size: 10
F1=100, D=2, F2=200, dropout_rate=0.25
timepoints: 20000
Time taken: ~ 15 minutes
Correct: 214, Total: 336
Correct A: 0, Total A: 122
Correct C: 214, Total C: 214
Accuracy: 63.6905%

Train 5: (Overfit to C)
Epochs: 300
Learning rate: 0.0007
Batch size: 10
F1=99, D=3, F2=201, dropout_rate=0.7
timepoints: 20000
Time taken: ~ 9 hours
Correct: 214, Total: 336
Correct A: 0, Total A: 122
Correct C: 214, Total C: 214
Accuracy: 63.6905%
"""