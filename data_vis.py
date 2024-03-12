import mne

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Load preprocessed data
raw = mne.io.read_raw_eeglab('eeg-data/sub-001/eeg/sub-001_task-eyesclosed_eeg.set', preload=False)
raw.plot()


# Wait for user key input to close the display
input("Press any key to close the display...")
