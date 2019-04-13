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
    def __init__(self, dn_parameters, input_parameters):
        # the main DenseNet model -- this function initializes the layers.
        super(DenseNetPerso, self).__init__()

        # Definition of the parameters
        self.dn_parameters = dn_parameters
        self.input_parameters = input_parameters

        """
        ##########################################################################
        ########## Initialisation of the weights of the Neural Network ###########
        ##########################################################################
        """

        """
        ------------------------------------------------------
        ---------- Initialisation of the audio part ----------
        ------------------------------------------------------
        """
        ##### First layer : ######
        # Conv 7
        self.nn_audio_firstLayer = nn.ModuleList([])
        self.nn_audio_firstLayer.append(
            nn.Conv1d(in_channels=self.input_parameters['audio']['nb_channels'],
                      out_channels=self.dn_parameters['audio']['k'], kernel_size=7, stride=1, padding=3)
        )
        # Max pooling
        self.nn_audio_firstLayer.append(
            nn.MaxPool1d(2, stride=1)
        )

        ##### Definition of the dense blocks ######
        self.nn_audio_denseBlock = nn.ModuleList([])
        k = self.dn_parameters['audio']['k']
        for b in range(self.dn_parameters['audio']['nb_blocks']):
            nb_layers = self.dn_parameters['audio']['nb_conv'][b]
            for conv in range(nb_layers):
                # Batch Normalization
                self.nn_audio_denseBlock.append(nn.BatchNorm1d(k * (conv+1)))
                # Activation Function
                """--> To do during forward computation"""
                # Conv
                self.nn_audio_denseBlock.append(
                    nn.Conv1d(in_channels=(k * (conv + 1)), out_channels=k, kernel_size=3, padding=1)
                )
                # Dropout
                self.nn_audio_denseBlock.append(nn.Dropout(0.2))
                # Then concatenation
                """--> To Do during forward computation"""

        ###### Definition of the dense transition block #####
        self.nn_audio_denseTransitionBlock = nn.ModuleList([])
        for b in range(1, self.dn_parameters['audio']['nb_blocks']):
            # Batch Normalization
            self.nn_audio_denseTransitionBlock.append(
                nn.BatchNorm1d(k * (self.dn_parameters['audio']['nb_conv'][b-1] + 1))
            )
            # Activation Function
            """ --> To do during forward computation"""
            # Conv
            self.nn_audio_denseTransitionBlock.append(
                nn.Conv1d(
                    in_channels=k * (self.dn_parameters['audio']['nb_conv'][b-1] + 1),
                    out_channels=self.dn_parameters['audio']['k'],
                    kernel_size=1,
                    padding=0
                )
            )
            # Dropout
            self.nn_audio_denseTransitionBlock.append(nn.Dropout(0.2))
            # Max pooling
            self.nn_audio_denseTransitionBlock.append(nn.MaxPool1d(2))

        ##### Definition of the last layer of the audio #####
        self.nn_audio_lastLayer = nn.ModuleList([])
        len_pooling = int(self.input_parameters['audio']['len'] / (self.dn_parameters['audio']['nb_blocks'] - 1))
        # Average pooling
        self.nn_audio_lastLayer.append(
            nn.AvgPool1d(len_pooling)
        )

        # x has to be flatten in ( -1, 2 * k * self.dn_parameters['audio']['nb_conv'][-1])
        # (still don't understand the '2')
        """ --> To do during forward computation"""

        # Fully Connected
        self.nn_audio_lastLayer.append(
            nn.Linear(
                k * (self.dn_parameters['audio']['nb_conv'][-1] + 1),
                self.dn_parameters['audio']['size_fc']
            )
        )
        # Activation Function
        """ --> To do during forward computation"""
        # Dropout
        self.nn_audio_lastLayer.append(nn.Dropout(0.2))


        """
        ---------------------------------------------------------
        ---------- Initialisation of the spectrum part ----------
        ---------------------------------------------------------
        """
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
                nn.BatchNorm2d(k * (self.dn_parameters['spectrum']['nb_conv'][b-1] + 1))
            )
            # Activation Function
            """--> To do during forward computation"""
            # Conv
            self.nn_spectrum_denseTransitionBlock.append(
                nn.Conv2d(
                    in_channels=k * (self.dn_parameters['spectrum']['nb_conv'][b-1] + 1),
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
        self.nn_spectrum_lastLayer = nn.ModuleList([])
        h_pooling = int(self.input_parameters['spectrum']['h'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1))
        w_pooling = int(self.input_parameters['spectrum']['w'] / (self.dn_parameters['spectrum']['nb_blocks'] - 1))
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
                k * (self.dn_parameters['spectrum']['nb_conv'][-1] + 1),
                self.dn_parameters['spectrum']['size_fc']
            )
        )
        # Activation Function
        """--> To do during forward computation"""
        # Dropout
        self.nn_spectrum_lastLayer.append(nn.Dropout(0.2))

        """
        ---------------------------------------------------------
        ---------- Initialisation of the features part ----------
        ---------------------------------------------------------
        """
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
                nn.BatchNorm1d(k * (self.dn_parameters['features']['nb_conv'][b - 1] + 1))
            )
            # Activation function
            """--> To do during forward computation"""
            # Conv
            self.nn_features_denseTransitionBlock.append(
                nn.Conv1d(
                    in_channels=k * (self.dn_parameters['features']['nb_conv'][b - 1] + 1),
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
        len_pooling = int(self.input_parameters['features']['len'] / (self.dn_parameters['features']['nb_blocks'] - 1))
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
                k * (self.dn_parameters['features']['nb_conv'][-1] + 1),
                self.dn_parameters['features']['size_fc']
            )
        )
        # Activation function
        """--> To do during forward computation"""
        # Dropout
        self.nn_features_lastLayer.append(nn.Dropout(0.2))

        """
        ------------------------------------------------------
        ---------- Initialisation of the fmstd part ----------
        ------------------------------------------------------
        """
        ##### Fully connected layers #####
        self.nn_fmstd_fc = nn.ModuleList([])
        for i in range(self.dn_parameters['fmstd']['nb_layers']):
            if i == 0:
                # Fully Connected
                self.nn_fmstd_fc.append(
                    nn.Linear(
                        self.input_parameters['fmstd']['len'],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            else:
                # Fully Connected
                self.nn_fmstd_fc.append(
                    nn.Linear(
                        self.dn_parameters['fmstd']['layers_size'][i-1],
                        self.dn_parameters['fmstd']['layers_size'][i]
                    )
                )
            """-- Append a relu in forward computation --"""
            self.nn_fmstd_fc.append(nn.Dropout(0.2))
        """
        ------------------------------------------------------
        ---------- Initialisation of the final part ----------
        ------------------------------------------------------
        """
        ##### Fully connected layers #####
        self.nn_final_fc = nn.ModuleList([])
        for i in range(self.dn_parameters['final']['nb_layers']):
            if i == 0:
                # Fully Connected
                self.nn_final_fc.append(
                    nn.Linear(
                        self.input_parameters['final']['len'],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            else:
                # Fully Connected
                self.nn_final_fc.append(
                    nn.Linear(
                        self.dn_parameters['final']['layers_size'][i-1],
                        self.dn_parameters['final']['layers_size'][i]
                    )
                )
            if i != self.dn_parameters['final']['nb_layers'] - 1:
                # Relu
                """ --> To do during forward computation"""
                # Dropout
                self.nn_final_fc.append(nn.Dropout(0.2))

    def forward(self, x_audio, x_spectrum, x_features, x_fmstd):
        # feed-forward propagation of the model. Here we have the inputs, which is propagated through the layers
        # x_spectrum has dimension (batch_size, channels, h = buffersSize/2 + 1, w=nbBuffers)
        # - for this model (16, 2, 1025, 431)
        # x_audio has dimension (batch_size, channels, audio len)
        # - for this model (16, 2, 240000)
        # x_features has dimension (batch_size, 2 * nbFeatures, w = nbBuffers)
        # - for this model (16, 2 * 5, 431)
        # x_fmstd has dimension (batch_size, 2 * 2 * nbFeatures)
        # - for this model (16, 1, 2 * 2 * 5)

        """
        -----------------------------------------------
        ---------- forward of the audio part ----------
        -----------------------------------------------
        """
        # Computation of the first part of the NN
        for f in self.nn_audio_firstLayer:
            x_audio = f(x_audio)

        # Computation of the DenseNet part
        i_denseBlock = 0
        i_denseTransitionBlock = 0
        for b in range(self.dn_parameters['audio']['nb_blocks']):
            nb_layers = self.dn_parameters['audio']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state_audio = x_audio
                x_audio = self.nn_audio_denseBlock[i_denseBlock](x_audio)  # Batch Normalization
                i_denseBlock += 1
                x_audio = F.relu(x_audio)
                x_audio = self.nn_audio_denseBlock[i_denseBlock](x_audio)  # Convolution
                i_denseBlock += 1
                x_audio = self.nn_audio_denseBlock[i_denseBlock](x_audio)  # Dropout
                i_denseBlock += 1
                x_audio = torch.cat((x_audio, previous_state_audio), dim=1)

            # Dense Transition Block
            if b != self.dn_parameters['audio']['nb_blocks'] - 1:
                x_audio = self.nn_audio_denseTransitionBlock[i_denseTransitionBlock](x_audio)  # Batch Normalization
                i_denseTransitionBlock += 1
                x_audio = F.relu(x_audio)
                x_audio = self.nn_audio_denseTransitionBlock[i_denseTransitionBlock](x_audio)  # Convolution
                i_denseTransitionBlock += 1
                x_audio = self.nn_audio_denseTransitionBlock[i_denseTransitionBlock](x_audio)  # Dropout
                i_denseTransitionBlock += 1
                x_audio = self.nn_audio_denseTransitionBlock[i_denseTransitionBlock](x_audio)  # Max Pooling
                i_denseTransitionBlock += 1

        # Computation of the last layer
        x_audio = self.nn_audio_lastLayer[0](x_audio)   # Average Pooling
        x_audio = x_audio.view(
            -1,
            self.dn_parameters['audio']['k'] * (self.dn_parameters['audio']['nb_conv'][-1] + 1)
        )
        x_audio = self.nn_audio_lastLayer[1](x_audio)   # Fully connected
        x_audio = F.relu(x_audio)
        x_audio = self.nn_audio_lastLayer[2](x_audio)   # Dropout


        """
        --------------------------------------------------
        ---------- forward of the spectrum part ----------
        --------------------------------------------------
        """
        # Computation of the first part of the NN
        for f in self.nn_spectrum_firstLayer:
            x_spectrum = f(x_spectrum)

        # Computation of the DenseNet part
        i_denseBlock = 0
        i_denseTransitionBlock = 0
        for b in range(self.dn_parameters['spectrum']['nb_blocks']):
            nb_layers = self.dn_parameters['spectrum']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state_spectrum = x_spectrum
                x_spectrum = self.nn_spectrum_denseBlock[i_denseBlock](x_spectrum)   # Batch Normalization
                i_denseBlock += 1
                x_spectrum = F.relu(x_spectrum)
                x_spectrum = self.nn_spectrum_denseBlock[i_denseBlock](x_spectrum)   # Convolution
                i_denseBlock += 1
                x_spectrum = self.nn_spectrum_denseBlock[i_denseBlock](x_spectrum)   # Dropout
                i_denseBlock += 1
                x_spectrum = torch.cat((x_spectrum, previous_state_spectrum), dim=1)

            # Dense Transition Block
            if b != self.dn_parameters['spectrum']['nb_blocks'] - 1:
                x_spectrum = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x_spectrum)    # Batch normalization
                i_denseTransitionBlock += 1
                x_spectrum = F.relu(x_spectrum)
                x_spectrum = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x_spectrum)    # Convolution
                i_denseTransitionBlock += 1
                x_spectrum = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x_spectrum)    # Dropout
                i_denseTransitionBlock += 1
                x_spectrum = self.nn_spectrum_denseTransitionBlock[i_denseTransitionBlock](x_spectrum)    # Max_spectrum pooling
                i_denseTransitionBlock += 1

        # Computation of the last layer
        x_spectrum = self.nn_spectrum_lastLayer[0](x_spectrum)  # Average pooling
        x_spectrum = x_spectrum.view(
            -1,
            self.dn_parameters['spectrum']['k'] * (self.dn_parameters['spectrum']['nb_conv'][-1] + 1)
        )
        x_spectrum = self.nn_spectrum_lastLayer[1](x_spectrum)    # Fully Connected
        x_spectrum = F.relu(x_spectrum)
        x_spectrum = self.nn_spectrum_lastLayer[2](x_spectrum)    # Dropout

        """
        --------------------------------------------------
        ---------- forward of the features part ----------
        --------------------------------------------------
        """
        # Computation of the first part of the NN
        for f in self.nn_features_firstLayer:
            x_features = f(x_features)

        # Computation of the DenseNet part
        i_denseBlock = 0
        i_denseTransitionBlock = 0
        for b in range(self.dn_parameters['features']['nb_blocks']):
            nb_layers = self.dn_parameters['features']['nb_conv'][b]
            # Dense Block
            for l in range(nb_layers):
                previous_state_features = x_features
                x_features = self.nn_features_denseBlock[i_denseBlock](x_features)  # Batch Normalization
                i_denseBlock += 1
                x_features = F.relu(x_features)
                x_features = self.nn_features_denseBlock[i_denseBlock](x_features)  # Convolution
                i_denseBlock += 1
                x_features = self.nn_features_denseBlock[i_denseBlock](x_features)  # Dropout
                i_denseBlock += 1
                x_features = torch.cat((x_features, previous_state_features), dim=1)

            # Dense Transition Block
            if b != self.dn_parameters['features']['nb_blocks'] - 1:
                x_features = self.nn_features_denseTransitionBlock[i_denseTransitionBlock](x_features)  # Batch Normalization
                i_denseTransitionBlock += 1
                x_features = F.relu(x_features)
                x_features = self.nn_features_denseTransitionBlock[i_denseTransitionBlock](x_features)  # Convolution
                i_denseTransitionBlock += 1
                x_features = self.nn_features_denseTransitionBlock[i_denseTransitionBlock](x_features)  # Dropout
                i_denseTransitionBlock += 1
                x_features = self.nn_features_denseTransitionBlock[i_denseTransitionBlock](x_features)  # Max_features Pooling
                i_denseTransitionBlock += 1

        # Computation of the last layer
        x_features = self.nn_features_lastLayer[0](x_features)   # Average Pooling
        x_features = x_features.view(
            -1,
            self.dn_parameters['features']['k'] * (self.dn_parameters['features']['nb_conv'][-1] + 1)
        )
        x_features = self.nn_features_lastLayer[1](x_features)   # Fully connected
        x_features = F.relu(x_features)
        x_features = self.nn_features_lastLayer[2](x_features)   # Dropout

        """
        -----------------------------------------------
        ---------- forward of the fmstd part ----------
        -----------------------------------------------
        """
        # Computation of the fully connected layers of the NN
        x_fmstd = x_fmstd.view(
            -1,
            self.input_parameters['fmstd']['len']
        )
        ifc = 0
        for i in range(self.dn_parameters['fmstd']['nb_layers']):
            x_fmstd = self.nn_fmstd_fc[ifc](x_fmstd)    # Fully connected
            ifc += 1
            x_fmstd = F.relu(x_fmstd)
            x_fmstd = self.nn_fmstd_fc[ifc](x_fmstd)
            ifc += 1
        """
        -----------------------------------
        ---------- Concatenation ----------
        -----------------------------------
        """
        x_final = x_spectrum
        x_final = torch.cat((x_final, x_audio), dim=1)
        x_final = torch.cat((x_final, x_features), dim=1)
        x_final = torch.cat((x_final, x_fmstd), dim=1)


        """
        -----------------------------------------------
        ---------- forward of the final part ----------
        -----------------------------------------------
        """
        # Computation of the fully connected layers of the NN
        ifc = 0
        for i in range(self.dn_parameters['final']['nb_layers']):
            x_final = self.nn_final_fc[ifc](x_final)    # Fully connected
            ifc += 1
            if i != self.dn_parameters['final']['nb_layers'] - 1:
                x_final = F.relu(x_final)
                x_final = self.nn_final_fc[ifc](x_final)    # Dropout
                ifc += 1

        """
        ----------------------------------------
        --------- Return of the value ----------
        ----------------------------------------
        """

        return F.log_softmax(x_final, dim=1)

