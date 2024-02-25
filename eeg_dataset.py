import os
import mne

from torch.utils.data import Dataset


def load_eeg_data(file_path):
    raw = mne.io.read_raw_eeglab(file_path)
    return raw.get_data()


class EEGDataset(Dataset):
    def __init__(self, data_directory, dataset):
        self.data_directory = data_directory
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_info = self.dataset[idx]
        file_path = os.path.join(self.data_directory, file_info['file_name'])

        # Load raw EEG data using MNE
        eeg_data = load_eeg_data(file_path)
        eeg_data = eeg_data.astype('float32')

        # Label
        label = 0 if file_info['label'] == 'A' else 1

        return eeg_data, label
