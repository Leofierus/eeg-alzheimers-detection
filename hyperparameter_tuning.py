import optuna
import json
import os
import random
import time
import torch
import warnings
import mne
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from eeg_net import EEGNet
from eeg_dataset import EEGDataset

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Set random seed
random.seed(42)


def objective(trial):
    start_time = time.time()
    # Define the search space for hyperparameters
    f1 = trial.suggest_categorical('F1', [5, 10, 19, 38, 47, 95])
    d = trial.suggest_int('D', 2, 7)
    f2 = f1 * d
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5, 0.6, 0.7])

    # Initialize the model with the suggested hyperparameters
    eegnet_model = EEGNet(num_channels=19, timepoints=1425, num_classes=3, F1=f1, D=d, F2=f2, dropout_rate=dropout_rate)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eegnet_model.to(device)

    # Define your loss function and optimizer
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    start_learning_rate = trial.suggest_float('start_learning_rate', 0.01, 0.1)
    optimizer = optim.Adam(eegnet_model.parameters(), lr=start_learning_rate)
    end_learning_rate = trial.suggest_float('end_learning_rate', 0.0001, 0.009)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                            end_factor=(end_learning_rate / start_learning_rate),
                                            total_iters=epochs * 5)

    validation_loss = []
    for epoch in range(epochs):
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

        scheduler.step()

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

        validation_loss.append(valid_loss)

    end_time = time.time()
    run_time = end_time - start_time
    print(f'Run time: {run_time:.3f}s')

    # Return the validation loss as the objective value to minimize
    return sum(validation_loss) / (len(validation_loss) * len(valid_dataloader))


data_dir = 'model-data'
data_file = 'labels.json'

with open(os.path.join(data_dir, data_file), 'r') as file:
    data_info = json.load(file)

temp_train_data = [d for d in data_info if d['type'] == 'train']

num_train_samples = int(0.2 * len(temp_train_data))
selected_train_data = random.choices(temp_train_data, k=num_train_samples)
# Separate training data by class
train_data_A = [d for d in selected_train_data if d['label'] == 'A']
train_data_C = [d for d in selected_train_data if d['label'] == 'C']
train_data_F = [d for d in selected_train_data if d['label'] == 'F']

# Determine the minimum number of samples for balancing
min_samples = min((len(train_data_A) + len(train_data_C)) / 2, (len(train_data_A) + len(train_data_F)) / 2,
                  (len(train_data_C) + len(train_data_F)) / 2)

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

# Randomly divide the training data into training and validation sets of ratio 80:20
num_train_samples = int(0.8 * len(balanced_train_data))
sample_train_data = random.choices(balanced_train_data, k=num_train_samples)
sample_val_data = [d for d in balanced_train_data if d not in sample_train_data]

train_dataset = EEGDataset(data_dir, sample_train_data)
valid_dataset = EEGDataset(data_dir, sample_val_data)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=20, shuffle=True)

# study = optuna.create_study(direction='minimize', study_name='eegnet_hyperparameter_tuning',
#                             storage='sqlite:///eegnet.db')
# study.optimize(objective, n_trials=30)

study = optuna.load_study(study_name='eegnet_hyperparameter_tuning', storage='sqlite:///eegnet.db')
study.optimize(objective, n_trials=50)
