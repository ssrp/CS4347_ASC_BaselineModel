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
        # Initialisation of the weights of the features part of the NN
        super(DenseNetPerso_final, self).__init__()

        ##### Fully connected layers #####
        self.nn['final']['fc'] = []
        for i in range(self.dn_parameters['final']['nb_layers']):
            if i == 0:
                # Fully Connected
                self.nn['final']['fc'].append(
                    nn.Linear(
                        self.input_parameters['final']['len'],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            else:
                # Fully Connected
                self.nn['final']['fc'].append(
                    nn.Linear(
                        self.dn_parameters['final']['layers_size'][i-1],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            if i != self.dn_parameters['final']['nb_layers'] - 1:
                # Relu
                self.nn['final']['fc'].append(F.relu)
                # Dropout
                self.nn['final']['fc'].append(nn.Dropout(0.2))

    def forward_final(self, x):
        # feed-forward propagation of the model.
        # x_spectrum has dimension (batch_size, concatenation of precedent layers)
        # - for this model (16, ?)

        # Computation of the fully connected layers of the NN
        for f in self.nn['final']['fc']:
            x = f(x)

        return x
