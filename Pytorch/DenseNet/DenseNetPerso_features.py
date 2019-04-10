from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class DenseNetPerso_features(nn.Module):
    def init_features(self):
        # Initialisation of the weights of the features part of the NN
        super(DenseNetPerso_features, self).__init__()

        ##### First layer : ######
        self.nn_features_firstLayer = nn.ModuleList([])
        # Conv 7
        self.nn_features_firstLayer.append(
            nn.Conv1d(in_channels=self.input_parameters['features']['nb_channels'],
                      out_channels=self.dn_parameters['features']['k'], kernel_size=7, stride=1,
                      padding=3)
        )
        # Max pooling
        self.nn_features_firstLayer.append(
            nn.MaxPool1d(2, stride=1)
        )

        ##### Definition of the dense blocks ######
        self.nn_features_denseBlock = nn.ModuleList([])
        k = self.dn_parameters['features']['k']
        for b in range(self.dn_parameters['features']['nb_blocks']):
            nb_layers = self.dn_parameters['features']['nb_conv'][b]
            for conv in range(nb_layers):
                # Batch Normalization
                self.nn_features_denseBlock.append(nn.BatchNorm1d(k * (conv + 1)))
                # Activation function
                """--> To do during forward computation"""
                # Conv
                self.nn_features_denseBlock.append(
                    nn.Conv1d(in_channels=(k * (conv + 1)), out_channels=k, kernel_size=3, padding=1)
                )
                # Dropout
                self.nn_features_denseBlock.append(nn.Dropout(0.2))
                # Then concatenation
                """--> To do during forward computation"""

        ###### Definition of the dense transition block #####
        self.nn_features_denseTransitionBlock = nn.ModuleList([])
        for b in range(1, self.dn_parameters['features']['nb_blocks']):
            # Batch Normalization
            self.nn_features_denseTransitionBlock.append(
                nn.BatchNorm1d(k * (self.dn_parameters['features']['nb_conv'][b - 1]))
            )
            # Activation function
            """--> To do during forward computation"""
            # Conv
            self.nn_features_denseTransitionBlock.append(
                nn.Conv1d(
                    in_channels=self.dn_parameters['features']['nb_conv'][b - 1],
                    out_channels=self.dn_parameters['features']['k'],
                    kernel_size=1,
                    padding=0
                )
            )
            # Dropout
            self.nn_features_denseTransitionBlock.append(nn.Dropout(0.2))
            # Max pooling
            self.nn_features_denseTransitionBlock.append(nn.MaxPool1d(2))

        ##### Definition of the last layer of the features
        self.nn_features_lastLayer = nn.ModuleList([])
        len_pooling = self.input_parameters['features']['len'] / (self.dn_parameters['features']['nb_blocks'] - 1)
        # Average pooling
        self.nn_features_lastLayer.append(
            nn.AvgPool1d(len_pooling)
        )

        # x has to be flatten in ( -1, 2 * k * self.dn_parameters['features']['nb_conv'][-1])
        # (still don't understand the '2')
        """--> To do during forward computation"""

        # Fully Connected
        self.nn_features_lastLayer.append(
            nn.Linear(
                2 * k * self.dn_parameters['features']['nb_conv'][-1],
                self.dn_parameters['features']['size_fc']
            )
        )
        # Activation function
        """--> To do during forward computation"""
        # Dropout
        self.nn_features_lastLayer.append(nn.Dropout(0.2))

    def forward_features(self, x):
        # feed-forward propagation of the model.
        # x_features has dimension (batch_size, 2 * nbFeatures, w = nbBuffers)
        # - for this model (16, 2 * 5, ?)

        # Computation of the first part of the NN
        for f in self.nn_features_firstLayer:
            x = f(x)

        # Computation of the DenseNet part
        i_denseBlock = 0
        i_denseTransitionBlock = 0
        for b in range(self.dn_parameters['features']['nb_blocks']):
            nb_layers = self.dn_parameters['features']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state = x
                x = self.nn_features_denseBlock[i_denseBlock](x)  # Batch Normalization
                i_denseBlock += 1
                x = F.relu(x)
                x = self.nn_features_denseBlock[i_denseBlock](x)  # Convolution
                i_denseBlock += 1
                x = self.nn_features_denseBlock[i_denseBlock](x)  # Dropout
                i_denseBlock += 1
                x = torch.cat((x, previous_state), dim=2)

            # Dense Transition Block
            if b != self.dn_parameters['features']['nb_blocks'] - 1:
                x = self.nn_features_denseTransitinoBlock[i_denseTransitionBlock](x)  # Batch Normalization
                i_denseTransitionBlock += 1
                x = F.relu(x)
                x = self.nn_features_denseTransitinoBlock[i_denseTransitionBlock](x)  # Convolution
                i_denseTransitionBlock += 1
                x = self.nn_features_denseTransitinoBlock[i_denseTransitionBlock](x)  # Dropout
                i_denseTransitionBlock += 1
                x = self.nn_features_denseTransitinoBlock[i_denseTransitionBlock](x)  # Max Pooling
                i_denseTransitionBlock += 1

        # Computation of the last layer
        x = self.nn_features_lastLayer[0](x)   # Average Pooling
        x.view(
            -1,
            2 * self.dn_parameters['features']['k'] * self.dn_parameters['features']['nb_conv'][-1]
        )
        x = self.nn_features_lastLayer[1](x)   # Fully connected
        x = F.relu(x)
        x = self.nn_features_lastLayer[2](x)   # Dropout

        return x
