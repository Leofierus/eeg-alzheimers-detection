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

model_file = 'eegNet.pth'
model = EEGNet(num_channels=19, timepoints=30000, num_classes=2, F1=64, D=2, F2=128, dropout_rate=0.5)
model.load_state_dict(torch.load(model_file))

data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

test_data = [d for d in data_info if d['type'] == 'test']
test_dataset = EEGDataset(data_dir, test_data)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

total_a = 0
total_c = 0
for entry in test_data:
    if entry['label'] == 'A':
        total_a += 1
    elif entry['label'] == 'C':
        total_c += 1

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
    for eeg_data, labels in test_dataloader:
        eeg_data, labels = eeg_data.to(device), labels.to(device)
        outputs = model(eeg_data)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs[:, 1].cpu().numpy())

        for i in range(labels.size(0)):
            if labels[i] == 0 and predicted[i] == 0:
                correct_a += 1
            elif labels[i] == 1 and predicted[i] == 1:
                correct_c += 1
            print(f'Predicted: {predicted[i]}, Model value: {outputs[i]}, Actual: {labels[i]}')

    print(f'\nCorrect: {correct}, Total: {total}')
    print(f'Correct A: {correct_a}, Total A: {total_a}')
    print(f'Correct C: {correct_c}, Total C: {total_c}')
    print(f'Accuracy: {100 * correct / total:.4f}%')

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('images/precision_recall_curve.png')
plt.close()
print("Precision-Recall Curve saved successfully")

# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('images/roc_curve.png')
plt.close()
print("ROC Curve saved successfully")

# F1 Score vs. Threshold Curve
f1_scores = [f1_score(all_labels, all_probs > threshold) for threshold in thresholds_pr]
plt.plot(thresholds_pr, f1_scores, label='F1 Score vs. Threshold Curve')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Threshold Curve')
plt.legend()
plt.savefig('images/f1_score_threshold_curve.png')
plt.close()
print("F1 Score vs. Threshold Curve saved successfully")

# AUC-ROC
roc_auc = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.plot(fpr, tpr, label=f'AUC-ROC = {roc_auc:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC Curve')
plt.legend()
plt.savefig('images/auc_roc_curve.png')
plt.close()
print("AUC-ROC Curve saved successfully")

# AUC-PR
pr_auc = average_precision_score(all_labels, all_probs)
plt.plot([0, 1], [np.sum(all_labels == 1) / len(all_labels)] * 2, 'k--', label='Random')
plt.plot(recall, precision, label=f'AUC-PR = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUC-PR Curve')
plt.legend()
plt.savefig('images/auc_pr_curve.png')
plt.close()
print("AUC-PR Curve saved successfully")

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_probs > 0.5)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['C', 'A'])
plt.yticks([0, 1], ['C', 'A'])
plt.savefig('images/confusion_matrix.png')
plt.close()
print("Confusion Matrix saved successfully")


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

Train 3:
Epochs: 180
Learning rate: 0.001
Batch size: 20
F1=64, D=2, F2=128, dropout_rate=0.5
timepoints: 30000
Time taken: 
"""