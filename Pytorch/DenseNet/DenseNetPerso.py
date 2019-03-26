from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DenseNetPerso(nn.Module):
    def __init__(self, parameters):
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DenseNetPerso, self).__init__()

        self.parameters = parameters

        ##### First layer : ######
        self.first_layer = []
        self.first_layer.append(
            nn.Conv2d(in_channels=2, out_channels=self.parameters['k'], kernel_size=7, stride=1)
        )
        self.first_layer.append(
            nn.MaxPool2d((2, 2), stride=1)
        )

        ##### Definition of the dense blocks ######
        self.dense_block = []
        k = parameters['k']
        for b in range(self.parameters['nb_blocks']):
            block = []
            nb_layers = self.parameters['nb_conv'][b]
            for conv in range(nb_layers):
                layer = []
                layer.append(nn.BatchNorm2d(k * (conv+1)))
                layer.append(F.relu)
                layer.append(
                    nn.Conv2d(in_channels=(k * (conv + 1)), out_channels=k, kernel_size=3, padding=1)
                )
                layer.append(nn.Dropout(0.2))
                # Then concatenation
                block.append(layer)
            self.dense_block.append(block)

        ###### Definition of the dense transition block #####
        self.dense_transition_block = []
        for b in range(1, self.parameters['nb_blocks']):
            block = []
            block.append(
                nn.BatchNorm2d(k * (self.parameters['nb_conv'][b-1]))
            )
            block.append(F.relu)
            block.append(
                nn.Conv2d(
                    in_channels=self.parameters['nb_conv'][b-1],
                    out_channels=self.parameters['k'],
                    kernel_size=1,
                    padding=0
                )
            )
            block.append(nn.Dropout(0.2))
            block.append(nn.MaxPool2d((2, 2)))
            self.dense_transition_block.append(block)

        ##### Definition of the last layer of the spectrum
        self.last_layers = []
        h_pooling = self.parameters['input_size']['h'] / self.parameters[]
        self.last_layers.append(
            nn.AvgPool2d((7, 7))
        )
        self.last_layers.append(

        )

    def forward(self, x):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        # Computation of the first part of the NN
        for f in self.first_layer:
            x = f(x)

        # Computation of the DenseNet part
        for b in range(self.parameters['nb_blocks']):
            nb_layers = self.parameters['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state = x
                for f in self.dense_block[b][l]:
                    x = f(x)
                x = torch.cat((x, previous_state), dim=3)

            # Dense Transition Block
            if b != self.parameters['nb_blocks'] - 1 :
                for f in self.dense_transition_block[b]:
                    x = f(x)
        # Computation of the last layer


        return True
