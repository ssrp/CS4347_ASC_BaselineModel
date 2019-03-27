from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DenseNetPerso_final(nn.Module):
    def init_final(self):
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DenseNetPerso_final, self).__init__()

        ##### Fully connected layers #####
        self.nn['final']['fc'] = []
        for i in range(self.dn_parameters['final']['nb_layers']):
            if i == 0:
                self.nn['final']['fc'].append(
                    nn.Linear(
                        self.input_parameters['final']['len'],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            else:
                self.nn['final']['fc'].append(
                    nn.Linear(
                        self.dn_parameters['final']['layers_size'][i-1],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            if i != self.dn_parameters['final']['nb_layers'] - 1:
                self.nn['final']['fc'].append(F.relu)
                self.nn['final']['fc'].append(nn.Dropout(0.2))

    def forward_final(self, x):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        # Computation of the fully connected layers of the NN
        for f in self.nn['final']['fc']:
            x = f(x)

        return x
