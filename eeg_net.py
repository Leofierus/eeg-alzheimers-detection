import torchviz
import mne
import torch
import torch.nn as nn



class EEGNet(nn.Module):
    def __init__(self, num_channels, timepoints, num_classes, F1, D, F2, dropout_rate):
        super(EEGNet, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.depthwise_conv1 = nn.Conv2d(F1, D * F1, kernel_size=(num_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(D * F1)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Block 2
        self.separable_conv2 = nn.Conv2d(D * F1, F2, kernel_size=(1, 16), padding=(0, 8))
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Classifier
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (timepoints // 32), 21)
        self.classifier = nn.Linear(21, num_classes)

    def forward(self, x):
        # Block 1
        # print(f'Shape (input): {x.shape}')
        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # Reshape
        # print(f'Shape (reshape): {x.shape}')
        x = self.conv1(x)
        # print(f'Shape (conv1): {x.shape}')
        x = self.batchnorm1(x)
        # print(f'Shape (batchnorm1): {x.shape}')
        x = self.depthwise_conv1(x)
        # print(f'Shape (depthwise_conv1): {x.shape}')
        x = self.batchnorm2(x)
        # print(f'Shape (batchnorm2): {x.shape}')
        x = self.activation1(x)
        # print(f'Shape (activation1): {x.shape}')
        x = self.avgpool1(x)
        # print(f'Shape (avgpool1): {x.shape}')
        x = self.dropout1(x)
        # print(f'Shape (dropout1): {x.shape}')

        # Block 2
        x = self.separable_conv2(x)
        # print(f'Shape (separable_conv2): {x.shape}')
        x = self.batchnorm3(x)
        # print(f'Shape (batchnorm3): {x.shape}')
        x = self.activation2(x)
        # print(f'Shape (activation2): {x.shape}')
        x = self.avgpool2(x)
        # print(f'Shape (avgpool2): {x.shape}')
        x = self.dropout2(x)
        # print(f'Shape (dropout2): {x.shape}')

        # Classifier
        x = self.flatten(x)
        # print(f'Shape (flatten): {x.shape}')
        x = self.dense(x)
        # print(f'Shape (dense): {x.shape}')
        x = self.classifier(x)
        # print(f'Shape (classifier): {x.shape}')

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def visualize_model(self, filename):
        x = torch.zeros(1, 19, 1425)
        y = self(x)
        g = torchviz.make_dot(y, params=dict(self.named_parameters()))
        g.render(filename, format='png', cleanup=True)
        return filename

    def visualize_temporal_filters(self, eeg_data, filename):
        channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'Fz', 'Cz', 'Pz']
        ch_dict = {str(i): channels[i] for i in range(19)}
        og_data = eeg_data
        eeg_data = eeg_data.view(eeg_data.size(0), 1, eeg_data.size(1), eeg_data.size(2))
        eeg_data = self.conv1(eeg_data)
        eeg_data = self.batchnorm1(eeg_data)

        np_x = eeg_data.detach().numpy()
        np_x = np_x.reshape(5, 19, 1426)
        og_data = og_data.detach().numpy()
        og_data = og_data.reshape(19, 1425)

        # Compute psds
        for i in range(5):
            eeg_raw = mne.io.RawArray(np_x[i], mne.create_info(ch_names=19, sfreq=95, ch_types='eeg'))
            eeg_raw.rename_channels(ch_dict)
            eeg_raw.set_montage('standard_1020')
            eeg_raw.compute_psd().plot(show=False).savefig(f'{filename}_{i}.png')
            eeg_raw.compute_psd().plot_topomap(show=False, size=10).savefig(f'{filename}_topomap_{i}.png')

        # Plot original data
        eeg_raw = mne.io.RawArray(og_data, mne.create_info(ch_names=19, sfreq=95, ch_types='eeg'))
        eeg_raw.rename_channels(ch_dict)
        eeg_raw.set_montage('standard_1020')
        eeg_raw.compute_psd().plot(show=False).savefig(f'{filename}_og.png')
        eeg_raw.compute_psd().plot_topomap(show=False, size=10).savefig(f'{filename}_topomap_og.png')


# # Example usage
# nc = 19
# ts = 30000
# ncl = 2  # A and C
# f1 = 64
# d = 2
# f2 = 16
# dr = 0.5
#
# model = EEGNet(nc, ts, ncl, f1, d, f2, dr)
# print(model)

# Model params
# num_chans = 19
# timepoints = 1425
# num_classes = 3
# F1 = 57
# D = 5
# F2 = 190
# dropout_rate = 0.5
#
# model_file = 'models/eegnet_5fold.pth'
# model = EEGNet(num_channels=num_chans, timepoints=timepoints, num_classes=num_classes, F1=F1, D=D,
#                F2=F2, dropout_rate=dropout_rate)
# model.load_state_dict(torch.load(model_file))
# print("Model loaded successfully")
#
# x = torch.zeros(1, 19, 1425)
# model.forward(x)

#
# # Visualize model
# model.visualize_model('images/eegnet_model')
# model.visualize_temporal_filters('images/eegnet_temporal_filters.png')
# model.visualize_spatial_filters('images/eegnet_spatial_filters.png')


# Model output
# Shape (input): torch.Size([1, 19, 1425])
# Shape (reshape): torch.Size([1, 1, 19, 1425])
# Shape (conv1): torch.Size([1, 57, 19, 1426])
# Shape (batchnorm1): torch.Size([1, 57, 19, 1426])
# Shape (depthwise_conv1): torch.Size([1, 285, 1, 1426])
# Shape (batchnorm2): torch.Size([1, 285, 1, 1426])
# Shape (activation1): torch.Size([1, 285, 1, 1426])
# Shape (avgpool1): torch.Size([1, 285, 1, 356])
# Shape (dropout1): torch.Size([1, 285, 1, 356])
# Shape (separable_conv2): torch.Size([1, 190, 1, 357])
# Shape (batchnorm3): torch.Size([1, 190, 1, 357])
# Shape (activation2): torch.Size([1, 190, 1, 357])
# Shape (avgpool2): torch.Size([1, 190, 1, 44])
# Shape (dropout2): torch.Size([1, 190, 1, 44])
# Shape (flatten): torch.Size([1, 8360])
# Shape (dense): torch.Size([1, 21])
# Shape (classifier): torch.Size([1, 3])
