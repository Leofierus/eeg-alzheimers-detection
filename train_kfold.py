import os
import json
import mne
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from eeg_net import EEGNet
from eeg_dataset import EEGDataset

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Set random seed
random.seed(42)

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

# Model
eegnet_model = EEGNet(num_channels=num_chans, timepoints=timepoints, num_classes=num_classes, F1=F1,
                      D=D, F2=F2, dropout_rate=dropout_rate)
print(eegnet_model)
print(f"Model params: num_channels={num_chans}, timepoints={timepoints}, num_classes={num_classes}, F1={F1}, "
      f"D={D}, F2={F2}, dropout_rate={dropout_rate}")
print(f'Trainable params: {eegnet_model.num_params()}\n')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eegnet_model.to(device)

# Data
data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

train_data = [d for d in data_info if d['type'] == 'train']

# Separate training data by class
train_data_A = [d for d in train_data if d['label'] == 'A']
train_data_C = [d for d in train_data if d['label'] == 'C']
train_data_F = [d for d in train_data if d['label'] == 'F']

# Determine the minimum number of samples for balancing
min_samples = min((len(train_data_A)+len(train_data_C))/2, (len(train_data_A)+len(train_data_F))/2,
                  (len(train_data_C)+len(train_data_F))/2)

a_index = int(min(min_samples, len(train_data_A)))
c_index = int(min(min_samples, len(train_data_C)))
f_index = int(min(min_samples, len(train_data_F)))

# Randomly sample from each class to create a balanced training set
balanced_train_data = (random.sample(train_data_A, a_index) +
                       random.sample(train_data_C, c_index) +
                       random.sample(train_data_F, f_index))

print(f'Before Balancing\nA: {len(train_data_A)}, C: {len(train_data_C)}, F: {len(train_data_F)}')
print(f'After Balancing\nA: {a_index}, C: {c_index}, F: {f_index}')
print(f'Total: {len(balanced_train_data)}')

# Create a new EEGDataset using the balanced training data
train_dataset = EEGDataset(data_dir, balanced_train_data)

# Initialize StratifiedKFold with 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Training loop
train_losses = []
epochs = 810
learning_rate = 0.01

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(eegnet_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, end_factor=0.001, total_iters=epochs*5)

for fold, (train_index, valid_index) in enumerate(skf.split(train_dataset.data, train_dataset.labels)):
    print(f'\nFold {fold + 1}/5')

    # Split data into train and validation sets for this fold
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    train_dataloader = DataLoader(train_dataset, batch_size=20, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=20, sampler=valid_sampler)

    print(f'Train dataloader: {len(train_dataloader)} batches')
    print(f'Valid dataloader: {len(valid_dataloader)} batches\n')

    for epoch in range(epochs):
        start_time = time.time()
        eegnet_model.train()
        train_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = eegnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)

        # Validation
        eegnet_model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in valid_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = eegnet_model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Fold {fold + 1}/5, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, '
              f'Valid Loss: {valid_loss:.6f}, Accuracy: {accuracy}, Time Taken: {epoch_time:.2f}s, '
              f'Learning Rate: {before_lr:.6f} -> {after_lr:.6f}')

# After training all folds, calculate average training loss
avg_train_loss = sum(train_losses) / len(train_losses)
print(f'Average Training Loss: {avg_train_loss}')

print('Training complete!')

# Save model
model_file = 'models/eegnet_5fold_train7.pth'
torch.save(eegnet_model.state_dict(), model_file)
print(f'Model saved to {model_file}')

# Plot losses and save plot
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold.png')
plt.close()

plt.plot(train_losses[:epochs], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold_fold1.png')
plt.close()

plt.plot(train_losses[epochs:2*epochs], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold_fold2.png')
plt.close()

plt.plot(train_losses[2*epochs:3*epochs], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold_fold3.png')
plt.close()

plt.plot(train_losses[3*epochs:4*epochs], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold_fold4.png')
plt.close()

plt.plot(train_losses[4*epochs:], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images_train12_kfold/train_losses_5fold_fold5.png')
plt.close()

print('Train losses plotted and saved!')
