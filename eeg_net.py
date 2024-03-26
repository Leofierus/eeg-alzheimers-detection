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
        self.classifier = nn.Linear(F2 * (timepoints // 32), num_classes)

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
        x = self.classifier(x)
        # print(f'Shape (classifier): {x.shape}')

        return x

    def predict(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv1(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)

        x = self.separable_conv2(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)

        x = self.flatten(x)
        x = self.classifier(x)

        return x


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
