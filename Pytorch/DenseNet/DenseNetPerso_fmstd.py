from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DenseNetPerso_fmstd(nn.Module):
    def init_fmstd(self):
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DenseNetPerso_fmstd, self).__init__()

        ##### Fully connected layers #####
        self.nn['fmstd']['fc'] = []
        for i in range(self.input_parameters['fmstd']['nb_layers']):
            if i == 0:
                self.nn['fmstd']['fc'].append(
                    nn.Linear(
                        self.input_parameters['fmstd']['len'],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            else:
                self.nn['fmstd']['fc'].append(
                    nn.Linear(
                        self.dn_parameters['fmstd']['layers_size'][i-1],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            self.nn['fmstd']['fc'].append(nn.Dropout(0.2))

    def forward_fmstd(self, x):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        # Computation of the fully connected layers of the NN
        for f in self.nn['fmstd']['fc']:
            x = f(x)

        return x
