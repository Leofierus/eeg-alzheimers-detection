import os
import json
import mne
import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from eeg_net import EEGNet
from eeg_dataset import EEGDataset

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

if not os.path.exists('images'):
    os.makedirs('images')

# Model
model = EEGNet(num_channels=19, timepoints=30000, num_classes=2, F1=64, D=2, F2=128, dropout_rate=0.5)
print(model)
print(f'\nArguments: num_channels=19, timepoints=30000, num_classes=2, F1=64, D=2, F2=128, dropout_rate=0.5\n')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Data
data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

train_data = [d for d in data_info if d['type'] == 'train']

test_data = []
count_a, count_c = 0, 0

for entry in data_info:
    if entry['type'] == 'test':
        if entry['label'] == 'A' and count_a < 10:
            test_data.append(entry)
            count_a += 1
        elif entry['label'] == 'C' and count_c < 10:
            test_data.append(entry)
            count_c += 1

train_dataset = EEGDataset(data_dir, train_data)
test_dataset = EEGDataset(data_dir, test_data)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Print train_dataloader info
print(f'Train dataset: {len(train_dataset)} samples')
print(f'Train dataloader: {len(train_dataloader)} batches')
print(f'Train dataloader batch size: {train_dataloader.batch_size}\n')

# Hyperparameters
learning_rate = 0.001
epochs = 180

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
test_losses = []
for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_dataloader)
    train_losses.append(average_loss)

    # Evaluation on the entire test set (single batch)
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(test_dataloader))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        average_test_loss = criterion(outputs, labels).item()
        test_losses.append(average_test_loss)

    end_time = time.time()
    epoch_time = end_time - start_time

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss}, Test Loss: {average_test_loss}'
          f', Time Taken: {epoch_time:.2f}s')


print('Training complete!')

# Plot losses and save plot
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/train_losses.png')
plt.close()

plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('images/test_losses.png')
plt.close()

print('Loss plots saved')

# Save model
model_file = 'eegNet.pth'
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}')
