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
        # Initialisation of the weights of the features part of the NN
        super(DenseNetPerso_fmstd, self).__init__()

        ##### Fully connected layers #####
        self.nn['fmstd']['fc'] = []
        for i in range(self.dn_parameters['fmstd']['nb_layers']):
            if i == 0:
                # Fully Connected
                self.nn['fmstd']['fc'].append(
                    nn.Linear(
                        self.input_parameters['fmstd']['len'],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            else:
                # Fully Connected
                self.nn['fmstd']['fc'].append(
                    nn.Linear(
                        self.dn_parameters['fmstd']['layers_size'][i-1],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            self.nn['fmstd']['fc'].append(F.relu)
            self.nn['fmstd']['fc'].append(nn.Dropout(0.2))

    def forward_fmstd(self, x):
        # feed-forward propagation of the model.
        # x_fmstd has dimension (batch_size, 2 * 2 * nbFeatures)
        # - for this model (16, 2 * 2 * 5)

        # Computation of the fully connected layers of the NN
        for f in self.nn['fmstd']['fc']:
            x = f(x)

        return x
