import torch
import os
import json
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from eeg_net import EEGNet
from eeg_dataset import EEGDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, roc_auc_score


warnings.filterwarnings('ignore', category=RuntimeWarning)

if not os.path.exists('images'):
    os.makedirs('images')

# Model params
num_chans = 19
timepoints = 1425
num_classes = 3
F1 = 5
D = 5
F2 = 25
dropout_rate = 0.5

model_file = 'models/eegnet_5fold_train7.pth'
model = EEGNet(num_channels=num_chans, timepoints=timepoints, num_classes=num_classes, F1=F1, D=D,
               F2=F2, dropout_rate=dropout_rate)
model.load_state_dict(torch.load(model_file))
print("Model loaded successfully")

data_dir = 'model-data'
data_file = 'labels.json'
data_type = 'test_cross'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

test_data = [d for d in data_info if d['type'] == data_type]
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
a_probs = []
c_probs = []
f_probs = []

# Test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    correct_a = 0
    a_as_c = 0
    a_as_f = 0
    correct_c = 0
    c_as_a = 0
    c_as_f = 0
    correct_f = 0
    f_as_a = 0
    f_as_c = 0
    for eeg_data, labels in tqdm(test_dataloader):
        eeg_data, labels = eeg_data.to(device), labels.to(device)
        outputs = model.forward(eeg_data)
        temp, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs.cpu().numpy())
        a_probs.extend(outputs[:, 0].cpu().numpy())
        c_probs.extend(outputs[:, 1].cpu().numpy())
        f_probs.extend(outputs[:, 2].cpu().numpy())

        for i in range(labels.size(0)):
            if labels[i] == 0:
                if predicted[i] == 0:
                    correct_a += 1
                elif predicted[i] == 1:
                    a_as_c += 1
                else:
                    a_as_f += 1
            elif labels[i] == 1:
                if predicted[i] == 1:
                    correct_c += 1
                elif predicted[i] == 0:
                    c_as_a += 1
                else:
                    c_as_f += 1
            else:
                if predicted[i] == 2:
                    correct_f += 1
                elif predicted[i] == 0:
                    f_as_a += 1
                else:
                    f_as_c += 1

accuracy = correct / total

confusion_matrix = np.zeros((3, 3))
confusion_matrix[0, 0] = correct_a
confusion_matrix[0, 1] = a_as_c
confusion_matrix[0, 2] = a_as_f
confusion_matrix[1, 0] = c_as_a
confusion_matrix[1, 1] = correct_c
confusion_matrix[1, 2] = c_as_f
confusion_matrix[2, 0] = f_as_a
confusion_matrix[2, 1] = f_as_c
confusion_matrix[2, 2] = correct_f

accuracy_ad_cn = (correct_a + correct_c) / (total_a + total_c)
accuracy_ftd_cn = (correct_c + correct_f) / (total_c + total_f)
accuracy_ad_ftd = (correct_a + correct_f) / (total_a + total_f)

precision_a = correct_a / (correct_a + a_as_c + a_as_f)
recall_a = correct_a / (correct_a + c_as_a + f_as_a)
f1_a = 2 * precision_a * recall_a / (precision_a + recall_a)
sensitivity_a = (correct_f + correct_c) / (correct_f + correct_c + f_as_a + c_as_a)

precision_c = correct_c / (correct_c + c_as_a + c_as_f)
recall_c = correct_c / (correct_c + a_as_c + f_as_c)
f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
sensitivity_c = (correct_a + correct_f) / (correct_a + correct_f + a_as_c + f_as_c)

precision_f = correct_f / (correct_f + f_as_a + f_as_c)
recall_f = correct_f / (correct_f + a_as_f + c_as_f)
f1_f = 2 * precision_f * recall_f / (precision_f + recall_f)
sensitivity_f = (correct_a + correct_c) / (correct_a + correct_c + a_as_f + c_as_f)

mAP = (precision_a + precision_c + precision_f) / 3
mAR = (recall_a + recall_c + recall_f) / 3
mF1 = (f1_a + f1_c + f1_f) / 3

print(f'\nCorrect: {correct}, Total: {total}')
print(f'Correct A: {correct_a}, A as C: {a_as_c}, A as F: {a_as_f}, Total A: {total_a}')
print(f'Correct C: {correct_c}, C as A: {c_as_a}, C as F: {c_as_f}, Total C: {total_c}')
print(f'Correct F: {correct_f}, F as A: {f_as_a}, F as C: {f_as_c}, Total F: {total_f}')
print(f'Accuracy: {100 * accuracy:.4f}%')
print(f'Accuracy for AD vs. CN: {100 * accuracy_ad_cn:.4f}%')
print(f'Accuracy for FTD vs. CN: {100 * accuracy_ftd_cn:.4f}%')
print(f'Accuracy for AD vs. FTD: {100 * accuracy_ad_ftd:.4f}%')
print(f'Precision A: {100 * precision_a:.4f}%, Recall A: {100 * recall_a:.4f}%, F1 A: {100 * f1_a:.4f}%, '
      f'Sensitivity A: {100 * sensitivity_a:.4f}%, Specificity A: {100 * recall_a:.4f}%')
print(f'Precision C: {100 * precision_c:.4f}%, Recall C: {100 * recall_c:.4f}%, F1 C: {100 * f1_c:.4f}%, '
      f'Sensitivity C: {100 * sensitivity_c:.4f}%, Specificity C: {100 * recall_c:.4f}%')
print(f'Precision F: {100 * precision_f:.4f}%, Recall F: {100 * recall_f:.4f}%, F1 F: {100 * f1_f:.4f}%, '
      f'Sensitivity F: {100 * sensitivity_f:.4f}%, Specificity F: {100 * recall_f:.4f}%')
print(f'mAP: {100 * mAP:.4f}%, mAR: {100 * mAR:.4f}%, mF1: {100 * mF1:.4f}%')

sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=['A', 'C', 'F'], yticklabels=['A', 'C', 'F'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

all_probs = np.array(all_probs)

# Add the min value of each row to itself to avoid negative values
for i in range(all_probs.shape[0]):
    all_probs[i] += abs(np.min(all_probs[i]))

all_probs = normalize(all_probs, axis=1, norm='l1')


fpr_a, tpr_a, _ = roc_curve(all_labels, a_probs, pos_label=0)
roc_auc_a, roc_auc_c, roc_auc_f = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
fpr_c, tpr_c, _ = roc_curve(all_labels, c_probs, pos_label=1)
fpr_f, tpr_f, _ = roc_curve(all_labels, f_probs, pos_label=2)


plt.plot(fpr_a, tpr_a, color='darkorange', lw=2, label=f'AUC A: {roc_auc_a:.6f}')
plt.plot(fpr_c, tpr_c, color='green', lw=2, label=f'AUC C: {roc_auc_c:.6f}')
plt.plot(fpr_f, tpr_f, color='red', lw=2, label=f'AUC F: {roc_auc_f:.6f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for OVR')
plt.legend()
plt.show()

"""
Simple Batch Training

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

Train 6: (Perfect fit to train)
Epochs: 300
Learning rate: 0.008, 0.001, 0.0007 (100 epochs each)
F1=152, D=5, F2=760, dropout_rate=0.5
timepoints: 7500
Time taken: ~ 3 days
Train stats:
Correct: 3700, Total: 3701
Correct A: 1590, Total A: 1591
Correct C: 1274, Total C: 1274
Correct F: 836, Total F: 836
Accuracy: 99.9730%
Test stats:
Correct: 524, Total: 909
Correct A: 214, Total A: 333
Correct C: 184, Total C: 319
Correct F: 126, Total F: 257
Accuracy: 57.6458%

NOTE: Added self.dense layer now, all models before this did not have this layer

5-Fold Cross Validation (Plots accidentally overwritten by the next train)
Epochs: 300/each fold
Learning rate: LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=epochs*5)
F1=57, D=5, F2=190, dropout_rate=0.5
timepoints: 1425
Time taken: ~ 1.5 days
Train stats:
Correct: 3603, Total: 3701
Correct A: 1493, Total A: 1591
Correct C: 1274, Total C: 1274
Correct F: 836, Total F: 836
Accuracy: 97.3521%
Test stats:
Correct: 671, Total: 909
Correct A: 183, Total A: 333
Correct C: 231, Total C: 319
Correct F: 257, Total F: 257
Accuracy: 73.8174%

NOTE: Divided into within and cross subject test data

5-Fold Cross Validation (Same model)
Epochs: 300/each fold
Learning rate: LinearLR(optimizer, start_factor=0.5, end_factor=0.001, total_iters=epochs*5)
F1=57, D=5, F2=190, dropout_rate=0.5
timepoints: 1425
Time taken: ~ 1.5 days
Train stats:
Correct: 3603, Total: 3701
Correct A: 1493, Total A: 1591
Correct C: 1274, Total C: 1274
Correct F: 836, Total F: 836
Accuracy: 97.3521%
Test Stats (Within):
Correct: 335, Total: 344
Correct A: 137, Total A: 146
Correct C: 126, Total C: 126
Correct F: 72, Total F: 72
Accuracy: 97.3837%
Test Stats (Cross):
Correct: 652, Total: 873
Correct A: 184, Total A: 319
Correct C: 221, Total C: 307
Correct F: 247, Total F: 247
Accuracy: 74.6850%

Skipping intermediate models, accuracy of around 78-85%

After Optuna training:

5-Fold Cross Validation
Epochs: 810/fold
Learning rate: LinearLR(optimizer, start_factor=0.9, end_factor=0.001, total_iters=epochs*5)
F1=5, D=5, F2=25, dropout_rate=0.5
timepoints: 1425
Time taken: ~ 2 days
Train stats:
Correct: 3212, Total: 3219
Correct A: 1383, Total A: 1388
Correct C: 1100, Total C: 1102
Correct F: 729, Total F: 729
Accuracy: 99.7825%
Test Stats (Within):
Correct: 338, Total: 344
Correct A: 141, Total A: 146
Correct C: 125, Total C: 126
Correct F: 72, Total F: 72
Accuracy: 98.2558%
Test Stats (Cross):
Correct: 806, Total: 873
Correct A: 293, Total A: 319
Correct C: 300, Total C: 307
Correct F: 213, Total F: 247
Accuracy: 92.3253%

"""