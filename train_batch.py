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
timepoints = 7500
num_classes = 3
F1 = 152
D = 5
F2 = 760
dropout_rate = 0.5

# Model
eegnet_model = EEGNet(num_channels=num_chans, timepoints=timepoints, num_classes=num_classes, F1=F1, D=D
                      , F2=F2, dropout_rate=dropout_rate)
print(eegnet_model)
print(f"Model params: num_channels={num_chans}, timepoints={timepoints}, num_classes={num_classes}, F1={F1}, "
      f"D={D}, F2={F2}, dropout_rate={dropout_rate}")
eegnet_model.load_state_dict(torch.load('models/eegNet_train6.pth'))

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

# Use SubsetRandomSampler to ensure balanced classes in DataLoader
indices = list(range(len(train_dataset)))
train_sampler = SubsetRandomSampler(indices)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)

# Print train_dataloader info
print(f'Train dataset: {len(train_dataset)} samples')
print(f'Train dataloader: {len(train_dataloader)} batches')
print(f'Train dataloader batch size: {train_dataloader.batch_size}\n')

# Hyperparameters
learning_rate = 0.0007
epochs = 100

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(eegnet_model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
test_losses = []
for epoch in range(epochs):
    start_time = time.time()
    eegnet_model.train()
    train_loss = 0.0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = eegnet_model(inputs)
        loss = criterion(outputs, labels)
        # print(f'Loss: {loss.item()}, Outputs: {outputs}, Labels: {labels}')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss)

    # # Evaluation on the entire test set (single batch)
    # eegnet_model.eval()
    # with torch.no_grad():
    #     inputs, labels = next(iter(test_dataloader))
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = eegnet_model(inputs)
    #     test_loss = criterion(outputs, labels).item()
    #     test_losses.append(test_loss)
    #
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Test Loss: , '
          f'Time Taken: {epoch_time:.2f}s')


print('Training complete!')

# Plot losses and save plot
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/train_losses.png')
plt.close()

print('Loss plots saved')

# Save model
model_file = 'eegNet.pth'
torch.save(eegnet_model.state_dict(), model_file)
print(f'Model saved to {model_file}')
