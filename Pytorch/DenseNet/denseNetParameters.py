####### Template of te parameters of the NN ######

dn_parameters = {
    'spectrum': {
        'k': 32,  # The number of channel in the denseNet
        'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
        'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
        'size_fc': 100  # Size of the fully connected at the end
    },
    'audio': {
        'k': 32,  # The number of channel in the denseNet
        'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
        'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
        'size_fc': 100  # Size of the fully connected at the end
    },
    'features': {
        'k': 32,  # The number of channel in the denseNet
        'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
        'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
        'size_fc': 100  # Size of the fully connected at the end
    }
}

input_parameters = {
    'spectrum': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2,  # The number of channels
        'h': None,  # The heigth of the input
        'w': None  # The width of the input
    },
    'audio': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2,  # The number of channels
        'len': None  # The length of the input
    },
    'features': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2,  # The number of channels
        'len': None  # The length of the input
    }
}
