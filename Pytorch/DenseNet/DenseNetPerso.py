from __future__ import print_function, division

import torch
import numpy as np

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.nn as nn

# Importation of personal classes
import Pytorch.DenseNet.DenseNetPerso_spectrum as dnp_s
import Pytorch.DenseNet.DenseNetPerso_audio as dnp_a

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DenseNetPerso(nn.Module, dnp_s.DenseNetPerso_spectrum, dnp_a.DenseNetPerso_audio):
    def __init__(self, dn_parameters, input_parameters):
        # the main CNN model -- this function initializes the layers. NOTE THAT we are not performing the conv/pooling
        # operations in this function (this is just the definition)
        super(DenseNetPerso, self).__init__()

        # Defintion of the parameters
        self.dn_parameters = dn_parameters
        self.input_parameters = input_parameters

        # Definition of the neural network
        self.nn = {
            'spectrum': {}
        }

        # Initialisation of the weights of the Neural Network
        self.init_spectrum()

    def forward(self, x_spectrum, x_audio):
        # feed-forward propagation of the model. Here we have the input x, which is propagated through the layers
        # x has dimension (batch_size, channels, mel_bins, time_indices) - for this model (16, 1, 40, 500)

        x_spectrum = self.forward_spectrum(x_spectrum)
        x = x_spectrum

        x_audio = self.forward_audio(x_audio)
        x = torch.cat((x, x_audio), dim=1)

        return x