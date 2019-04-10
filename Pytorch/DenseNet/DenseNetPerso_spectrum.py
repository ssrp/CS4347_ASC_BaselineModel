from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DenseNetPerso_spectrum(nn.Module):
    def init_spectrum(self):
        # Initialisation of the weights of the features part of the NN
        super(DenseNetPerso_spectrum, self).__init__()

        ##### First layer : ######
        self.nn_spectrum_firstLayer = nn.ModuleList([])
        # Conv 7
        self.nn_spectrum_firstLayer.append(
            nn.Conv2d(in_channels=self.input_parameters['spectrum']['nb_channels'], out_channels=self.dn_parameters['spectrum']['k'], kernel_size=7, stride=1,
                      padding=3)
        )
        # Max Pooling
        self.nn_spectrum_firstLayer.append(
            nn.MaxPool2d((2, 2), stride=1)
        )

        ##### Definition of the dense blocks ######
        self.nn_spectrum_denseBlock = nn.ModuleList([])
        k = self.dn_parameters['spectrum']['k']
        for b in range(self.dn_parameters['spectrum']['nb_blocks']):
            nb_layers = self.dn_parameters['spectrum']['nb_conv'][b]
            for conv in range(nb_layers):
                # Batch Normalization
                self.nn_spectrum_denseBlock.append(nn.BatchNorm2d(k * (conv+1)))
                # Activation Function
                """ --> To do during forward computation"""
                # Convolution
                self.nn_spectrum_denseBlock.append(
                    nn.Conv2d(in_channels=(k * (conv + 1)), out_channels=k, kernel_size=3, padding=1)
                )
                # Dropout
                self.nn_spectrum_denseBlock.append(nn.Dropout(0.2))
                # Then concatenation
                """--> To do during forward computation"""

        ###### Definition of the dense transition block #####
        self.nn_spectrum_denseTransitionBlock = nn.ModuleList([])
        for b in range(1, self.dn_parameters['spectrum']['nb_blocks']):
            # Batch Normalization
            self.nn_spectrum_denseTransitionBlock.append(
                nn.BatchNorm2d(k * (self.dn_parameters['spectrum']['nb_conv'][b-1]))
            )
            # Activation Function
            """--> To do during forward computation"""
            # Conv
            self.nn_spectrum_denseTransitionBlock.append(
                nn.Conv2d(
                    in_channels=self.dn_parameters['spectrum']['nb_conv'][b-1],
                    out_channels=self.dn_parameters['spectrum']['k'],
                    kernel_size=1,
                    padding=0
                )
            )
            # Dropout
            self.nn_spectrum_denseTransitionBlock.append(nn.Dropout(0.2))
            # Max Pooling
            self.nn_spectrum_denseTransitionBlock.append(nn.MaxPool2d((2, 2)))

        ##### Definition of the last layer of the spectrum
        self.nn_spectrum_lastLayer = []
        h_pooling = self.input_parameters['spectrum']['h'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1)
        w_pooling = self.input_parameters['spectrum']['w'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1)
        # Average Pooling
        self.nn_spectrum_lastLayer.append(
            nn.AvgPool2d((h_pooling, w_pooling))
        )

        # x has to be flatten in ( -1, 2 * k * self.dn_parameters['spectrum']['nb_conv'][-1])
        # (still don't understand the '2')
        """--> To do during forward computation"""

        # Fully Connected
        self.nn_spectrum_lastLayer.append(
            nn.Linear(
                2 * k * self.dn_parameters['spectrum']['nb_conv'][-1],
                self.dn_parameters['spectrum']['size_fc']
            )
        )
        # Activation Function
        """--> To do during forward computation"""
        # Dropout
        self.nn_spectrum_lastLayer.append(nn.Dropout(0.2))

    def forward_spectrum(self, x):
        # feed-forward propagation of the model.
        # x_spectrum has dimension (batch_size, channels, h = buffersSize/2 + 1, w=nbBuffers)
        # - for this model (16, 2, ?, ?)

        # Computation of the first part of the NN
        for f in self.nn_spectrum_firstLayer:
            x = f(x)

        # Computation of the DenseNet part
        i_denseBlock = 0
        i_denseTransitionBlock = 0
        for b in range(self.dn_parameters['spectrum']['nb_blocks']):
            nb_layers = self.dn_parameters['spectrum']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state = x
                x = self.nn_spectrum_denseBlock[i_denseBlock](x)   # Batch Normalization
                i_denseBlock += 1
                x = F.relu(x)
                x = self.nn_spectrum_denseBlock[i_denseBlock](x)   # Convolution
                i_denseBlock += 1
                x = self.nn_spectrum_denseBlock[i_denseBlock](x)   # Dropout
                i_denseBlock += 1
                x = torch.cat((x, previous_state), dim=3)

            # Dense Transition Block
            if b != self.dn_parameters['spectrum']['nb_blocks'] - 1:
                x = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x)    # Batch normalization
                i_denseTransitionBlock += 1
                x = F.relu(x)
                x = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x)    # Convolution
                i_denseTransitionBlock += 1
                x = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x)    # Dropout
                i_denseTransitionBlock += 1
                x = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x)    # Max pooling
                i_denseTransitionBlock += 1

        # Computation of the last layer
        x = self.nn_spectrum_lastLayer[0](x)    # Average pooling
        x.view(
            -1,
            2 * self.dn_parameters['spectrum']['k'] * self.dn_parameters['spectrum']['nb_conv'][-1]
        )
        x = self.nn_spectrum_lastLayer[0](x)    # Fully Connected
        x = F.relu(x)
        x = self.nn_spectrum_lastLayer[0](x)    # Dropout
