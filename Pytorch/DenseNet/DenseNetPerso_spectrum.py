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
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DenseNetPerso_spectrum, self).__init__()

        ##### First layer : ######
        self.nn['spectrum']['first_layer'] = []
        self.nn['spectrum']['first_layer'].append(
            nn.Conv2d(in_channels=self.input_parameters['spectrum']['nb_channels'], out_channels=self.dn_parameters['spectrum']['k'], kernel_size=7, stride=1,
                      padding=3)
        )
        self.nn['spectrum']['first_layer'] .append(
            nn.MaxPool2d((2, 2), stride=1)
        )

        ##### Definition of the dense blocks ######
        self.nn['spectrum']['dense_block'] = []
        k = self.dn_parameters['spectrum']['k']
        for b in range(self.dn_parameters['spectrum']['nb_blocks']):
            block = []
            nb_layers = self.dn_parameters['spectrum']['nb_conv'][b]
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
            self.nn['spectrum']['dense_block'].append(block)

        ###### Definition of the dense transition block #####
        self.nn['spectrum']['dense_transition_block'] = []
        for b in range(1, self.dn_parameters['spectrum']['nb_blocks']):
            block = []
            block.append(
                nn.BatchNorm2d(k * (self.dn_parameters['spectrum']['nb_conv'][b-1]))
            )
            block.append(F.relu)
            block.append(
                nn.Conv2d(
                    in_channels=self.dn_parameters['spectrum']['nb_conv'][b-1],
                    out_channels=self.dn_parameters['spectrum']['k'],
                    kernel_size=1,
                    padding=0
                )
            )
            block.append(nn.Dropout(0.2))
            block.append(nn.MaxPool2d((2, 2)))
            self.nn['spectrum']['dense_transition_block'].append(block)

        ##### Definition of the last layer of the spectrum
        self.nn['spectrum']['last_layers'] = []
        h_pooling = self.input_parameters['spectrum']['h'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1)
        w_pooling = self.input_parameters['spectrum']['w'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1)
        self.nn['spectrum']['last_layers'].append(
            nn.AvgPool2d((h_pooling, w_pooling))
        )

        # x has to be flatten in ( -1, 2 * k * self.dn_parameters['spectrum']['nb_conv'][-1]) (still don't understand the '2')

        self.nn['spectrum']['last_layers'].append(
            nn.Linear(
                2 * k * self.dn_parameters['spectrum']['nb_conv'][-1],
                self.dn_parameters['spectrum']['size_fc']
            )
        )
        self.nn['spectrum']['last_layers'].append(nn.Dropout(0.2))

    def forward_spectrum(self, x):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        # Computation of the first part of the NN
        for f in self.nn['spectrum']['first_layer']:
            x = f(x)

        # Computation of the DenseNet part
        for b in range(self.dn_parameters['spectrum']['nb_blocks']):
            nb_layers = self.dn_parameters['spectrum']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state = x
                for f in self.nn['spectrum']['dense_block'][b][l]:
                    x = f(x)
                x = torch.cat((x, previous_state), dim=3)

            # Dense Transition Block
            if b != self.dn_parameters['spectrum']['nb_blocks'] - 1 :
                for f in self.nn['spectrum']['dense_transition_block'][b]:
                    x = f(x)
        # Computation of the last layer
        i = 0
        for f in self.nn['spectrum']['last_layer']:
            x = f(x)
            if i == 0:
                x.view(
                    -1,
                    2 * self.dn_parameters['spectrum']['k'] * self.dn_parameters['spectrum']['nb_conv'][-1]
                )

        return x
