from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DN_Spectrum(nn.Module):
    def __init__(self):
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DN_Spectrum, self).__init__()

        # first conv layer, extracts 32 feature maps from 1 channel input
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        # batch normalization layer
        self.conv1_bn = nn.BatchNorm2d(32)
        # max pooling of 5x5
        self.mp1 = nn.MaxPool2d((5, 5))
        # dropout layer, for regularization of the model (efficient learning)
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d((4, 100))
        self.drop2 = nn.Dropout(0.3)
        # a dense layer
        self.fc1 = nn.Linear(1 * 2 * 64, 100)
        self.drop3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 10)

    def denseBlock2D(self):
        return None

    def forward(self, x):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        # perfrom first convolution
        x = self.conv1(x)
        # batch normalization
        x = self.conv1_bn(x)
        # ReLU activation
        x = F.relu(x)

        # Max pooling, results in 32 8x100 feature maps [output -> (16, 32, 8, 100)]
        x = self.mp1(x)

        # apply dropout
        x = self.drop1(x)

        # next convolution layer (results in 64 feature maps) output: (16, 64, 4, 100)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)

        # max pooling of 4, 100. Results in 64 2x1 feature maps (16, 64, 2, 1)
        x = self.mp2(x)
        x = self.drop2(x)

        # Flatten the layer into 64x2x1 neurons, results in a 128D feature vector (16, 128)
        x = x.view(-1, 1 * 2 * 64)

        # add a dense layer, results in 100D feature vector (16, 100)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop3(x)

        # add the final output layer, results in 10D feature vector (16, 10)
        x = self.fc2(x)

        # add log_softmax for the label
        return F.log_softmax(x, dim=1)
