####### Template of te parameters of the NN ######

dn_parameters = [
    {
        'name': 'small',
        'spectrum': {
            'k': 8,  # The number of channel in the denseNet
            'nb_blocks': 2,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [3, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 20  # Size of the fully connected at the end
        },
        'audio': {
            'k': 8,  # The number of channel in the denseNet
            'nb_blocks': 2,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 15  # Size of the fully connected at the end
        },
        'features': {
            'k': 6,  # The number of channel in the denseNet
            'nb_blocks': 2,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 10,  # Size of the fully connected at the end
        },
        'fmstd': {
            'nb_layers': 2,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [20, 10],  # The size of the layers in fully connected layers
        },
        'final': {  # The parameters for the fully connected layers at the end of the neural network
            'nb_layers': 2,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [30, 10],  # The size of the layers in the fully connected layers
        }
    },      # Small
    {
        'name': 'medium',
        'spectrum': {
            'k': 16,  # The number of channel in the denseNet
            'nb_blocks': 3,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 3, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 50  # Size of the fully connected at the end
        },
        'audio': {
            'k': 12,  # The number of channel in the denseNet
            'nb_blocks': 3,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 3, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 20  # Size of the fully connected at the end
        },
        'features': {
            'k': 8,  # The number of channel in the denseNet
            'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [3, 3, 2, 2],  # The numbers of convolutional layers in a dense block
            'size_fc': 15,  # Size of the fully connected at the end
        },
        'fmstd': {
            'nb_layers': 3,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [40, 20, 10],  # The size of the layers in fully connected layers
        },
        'final': {  # The parameters for the fully connected layers at the end of the neural network
            'nb_layers': 2,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [40, 10],  # The size of the layers in the fully connected layers
        }
    },      # Medium
    {
        'name': 'big',
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
            'size_fc': 50,  # Size of the fully connected at the end
        },
        'fmstd': {
            'nb_layers': 3,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [80, 50, 20],  # The size of the layers in fully connected layers
        },
        'final': {  # The parameters for the fully connected layers at the end of the neural network
            'nb_layers': 3,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [80, 40, 10],  # The size of the layers in the fully connected layers
        }
    }       # Big
]


def name2indice(name):
    for i in range(len(dn_parameters)):
        if dn_parameters[i]['name'] == name:
            return i
    return None


def returnSmallModel():
    indice = name2indice('small')
    return dn_parameters[indice]


def return_model_parameters(arg):
    if isinstance(arg, int):
        i = arg
        return dn_parameters[i]
    elif isinstance(arg, str):
        name = arg
        indice = name2indice(name)
        if indice:
            return dn_parameters[indice]
        else:
            return returnSmallModel()
    else:
        return returnSmallModel()

input_parameters = {
    'spectrum': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2,  # The number of channels (right and left)
        'h': None,  # The heigth of the input (bufferLength/2 + 1)
        'w': None  # The width of the input (nbBuffers)
    },
    'audio': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2,  # The number of channels (right and left)
        'len': None  # The length of the input
    },
    'features': {
        'batch_size': 16,  # The size of each batch
        'nb_channels': 2 * 5,  # The number of channels (nbFeatures * 2) (because right and left)
        'len': None  # The length of the input (nbBuffers)
    },
    'fmstd': {
        'len': 2 * 2 * 5  # The size of the input (nbFeatures * 2 * 2) (because (right and left) and (mean and variance)
    },
    'final': {
        'len': 180  # The input size = spectrum size_fc + audio size_fc + features size_fc + fmstd layers_size[-1]
    }
}

def return_input_parameters():
    return input_parameters