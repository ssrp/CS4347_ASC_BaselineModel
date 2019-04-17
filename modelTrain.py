from __future__ import print_function, division

import argparse
import os
# Ignore warnings
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# import PyTorch Functionalities
import torch.optim as optim

# Personal imports
from DataGeneration import inputGeneration as ig
import DataGeneration.dataNormalization as dn
from DataGeneration.DCASEDataset import DCASEDataset
from Pytorch.DenseNet.DenseNetPerso import DenseNetPerso
import Pytorch.DenseNet.denseNetParameters as DN_param
import Pytorch.useModel as useModel

"""
    This file is used to (load) train a model
"""


def main():
    """
        main function which will do everything to train the model
    """
    # Training settings
    parser = argparse.ArgumentParser(description='Can be use do a train for the ASC project (CS4347)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # Personal arguments
    parser.add_argument('--light-train', action='store_true', default=False,
                        # If we want to work on a small number of training data (for test on CPU)
                        help='For training on a small number of data')
    parser.add_argument('--light-test', action='store_true', default=False,
                        # If we want to work on a small number of testing data (for test on CPU)
                        help='For testing on a small number of data')
    parser.add_argument('--light-data', action='store_true', default=False,
                        # If we want to work on a small number of training and testing data (for test on CPU)
                        help='--light-train & --light-test')
    parser.add_argument('--light-all', action='store_true', default=False,
                        # If we want to work on a small number of training and testing data and a small NN model
                        # (for test on CPU)
                        help='--light-data & small model')
    parser.add_argument('--name', default='',  # Optional, if we want ot name our model
                        help='The name of the model')
    parser.add_argument('--model-id', default='big',  # the id of the model (the name in dn_parameters)
                        help='Model ID')
    parser.add_argument('--inputs-used', default='1111',
                        # The inputs used (1=used, 0=not used) (waveform, spectrum, features, fmstd)
                        help='The inputs used (1=used, 0=not used) (waveform, spectrum, features, fmstd)')
    parser.add_argument('--load-model', default='',  # optional : if we want to load a model
                        help='Load the model modelName-inputsUsed(-name)-nbEpochs-idx')

    # CUDA
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # init the train and test directories
    train_labels_dir = 'Dataset/train/train_labels.csv'
    test_labels_dir = 'Dataset/test/test_labels.csv'
    train_data_dir = 'Dataset/train/'
    test_data_dir = 'Dataset/test/'

    ##### Creation of the folders for the Generated Dataset #####

    # The arguments 'light' are made to test the network on my poor little computer and its poor little CPU
    if args.light_all:
        args.model_id = 'small'
    light_train = args.light_all or args.light_data or args.light_train  # Only 5 files in the train dataset
    light_test = args.light_all or args.light_data or args.light_test  # Only 5 files in the test dataset

    # The parameters of the DenseNet Model
    dn_parameters = DN_param.return_model_parameters(args.model_id)

    # Creation of the folder where we gonna save the computed inputs
    if light_train:
        # If we want to test on a little CPU
        ig.setLightEnviromnent()
        g_train_data_dir = './GeneratedLightDataset/train/'
        g_test_data_dir = './GeneratedLightDataset/test/'
        g_data_dir = './GeneratedLightDataset/'
    else:
        # Made for training on GPU (big dataset size)
        ig.setEnviromnent()
        g_train_data_dir = './GeneratedDataset/train/'
        g_test_data_dir = './GeneratedDataset/test/'
        g_data_dir = './GeneratedDataset/'

    # Creation of the variables 'normalization_values' and 'input_parameters'
    model_loaded = args.load_model != ''
    if not model_loaded:  # If we don't load a model
        if os.path.isfile(os.path.join(g_data_dir, 'normalization_values.npy')) \
                and os.path.isfile(os.path.join(g_data_dir, 'input_parameters.npy')):
            # get the mean and std. If Normalized already, just load the npy files and comment
            #  the NormalizeData() function above
            normalization_values = np.load(os.path.join(g_data_dir, 'normalization_values.npy')).item()
            input_parameters = np.load(os.path.join(g_data_dir, 'input_parameters.npy')).item()
            print(
                'LOAD OF THE FILE normalization_values.npy FOR NORMALIZATION AND input_parameters.npy FOR THE NEURAL NETWORK')

        else:
            # If not, run the normalization and save the mean/std
            print('DATA NORMALIZATION : ACCUMULATING THE DATA')
            # Get the normalized values
            normalization_values = dn.NormalizeData(
                train_labels_dir=os.path.abspath(train_labels_dir),
                root_dir=os.path.abspath(train_data_dir),
                save_dir=os.path.abspath(g_train_data_dir),
                light_data=light_train
            )
            # Save them
            np.save(os.path.join(g_data_dir, 'normalization_values.npy'), normalization_values)
            # Create the informations of the inputs
            ig.createInputParametersFile(
                template=DN_param.input_parameters,
                fileName=os.path.abspath(os.path.join(g_data_dir, 'input_parameters.npy')),
                dn_parameters=dn_parameters
            )
            # Save them
            input_parameters = np.load(os.path.join(g_data_dir, 'input_parameters.npy')).item()
            print('DATA NORMALIZATION COMPLETED')
    else:       # We load the model
        folder_id = args.load_model.split('-')
        args.model_id = folder_id[0]  # small / big / medium
        hasName = len(folder_id) == 5  # We check if we put a name on our loaded model
        nom = '_Name({0})'.format(folder_id[2]) if hasName else ''
        b = 1 if hasName else 0
        folder_name = 'Model({0})_InputsUsed({1}){2}_NbEpochs({3})_({4})'.format(  # The folder where the model is saved
            args.model_id,           # small, big, medium
            folder_id[1],       # inputs used
            nom,                # name (optional)
            folder_id[2+b],     # nb epochs
            folder_id[3+b]      # index
        )
        model_folder = 'SavedModels/{0}/{1}'.format(folder_id[0], folder_name)
        summaryDictLoaded = np.load(  # We load all the usable information through this dictionnary
            os.path.join(model_folder, folder_name + '.npy')
        ).item()
        dn_parameters = summaryDictLoaded['dn_parameters']  # The parameters of the DenseNet Model
        input_parameters = summaryDictLoaded['input_parameters']  # The information on the shape of the inputs
        normalization_values = summaryDictLoaded['normalization_values']  # The values to normalize the inputs

        args.inputs_used = summaryDictLoaded['inputs_used']
        if args.name == '' and hasName:
            args.name = folder_id[2]

    # Transformers to normalize the inputs at the entrance of the neural network
    data_transform = dn.return_data_transform(normalization_values)

    # init the datasets
    dcase_dataset = DCASEDataset(
        csv_file=train_labels_dir,
        root_dir=train_data_dir,
        save_dir=g_train_data_dir,
        transform=data_transform,
        light_data=light_train)
    dcase_dataset_test = DCASEDataset(
        csv_file=test_labels_dir,
        root_dir=test_data_dir,
        save_dir=g_test_data_dir,
        transform=data_transform,
        light_data=light_test)

    # set number of cpu workers in parallel
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    # Get the training and testing data loader
    train_loader = torch.utils.data.DataLoader(
        dcase_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dcase_dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs)

    # init the model
    model = DenseNetPerso(
        dn_parameters=dn_parameters,
        input_parameters=input_parameters,
        inputs_used=args.inputs_used
    ).to(device)

    # Load the model if it is asked
    if model_loaded:
        model.load_state_dict(torch.load(os.path.join(model_folder, folder_name + '.pt')))
        print('Model {0} loaded'.format(args.load_model))


    # init the optimizer
    optimizer_adam = optim.Adam(model.parameters(),     # Documentation says it is better with SGD, but SGD optimizer
                                lr=args.lr)             # didn't perform well when we tried

    print('MODEL TRAINING START')
    # Prepare the training
    loss_train = summaryDictLoaded['loss_train'] if model_loaded else []
    acc_train = summaryDictLoaded['acc_train'] if model_loaded else []
    loss_test = summaryDictLoaded['loss_test'] if model_loaded else []
    acc_test = summaryDictLoaded['acc_test'] if model_loaded else []

    # Create the architecture of the saved models if it is not already done
    if not os.path.isdir('./SavedModels'):      # One big folder "SavedModels"
        os.mkdir('./SavedModels')
    model_name = DN_param.id2name(args.model_id)
    model_folder = os.path.join('./SavedModels', model_name)
    if not os.path.isdir(model_folder):         # One subfolder for each DenseNet architecture
        os.mkdir(model_folder)
    # Find a name for the save because we don't want any conflict
    nom = '' if args.name == '' else '_Name({0})'.format(args.name)
    flag = True
    i = 0
    bias_epoch = 0 if args.load_model == '' else summaryDictLoaded[
        'nb_epochs']  # bias nb epochs (the number of epochs the models has already been trained on
    while flag:     # Look for a name that has't been choosen before to prevent conflic
        all_name = 'Model({0})_InputsUsed({1}){2}_NbEpochs({3})_({4})'.format(
            model_name,
            args.inputs_used,
            nom,
            args.epochs + bias_epoch,
            i,
        )
        folder_path = os.path.join(model_folder, all_name)
        if not os.path.isdir(folder_path):  # We've found our named !!
            flag = False
        i += 1
    os.mkdir(folder_path)

    # The result of the model with the best test accuracy
    best_acc_test = 0 if not model_loaded else summaryDictLoaded['best_model']['acc_test']
    b_a_train = 0 if not model_loaded else summaryDictLoaded['best_model']['acc_train']
    b_l_test = 0 if not model_loaded else summaryDictLoaded['best_model']['loss_test']
    b_l_train = 0 if not model_loaded else summaryDictLoaded['best_model']['loss_train']
    best_epoch = 0 if not model_loaded else summaryDictLoaded['best_model']['epoch']

    # We save it a 1st time in case the model won't get better after because it is saved only when the test accuracy is
    #  the highest reashed (could never be the case if the network has been loaded
    torch.save(model.state_dict(), os.path.join(folder_path, all_name + '.pt'))

    # train the model
    for epoch in range(1 + bias_epoch, args.epochs + 1 + bias_epoch):
        useModel.train(args, model, device, train_loader, optimizer_adam, epoch)
        l_train, a_train = useModel.test(args, model, device, train_loader, 'Training Data')
        l_test, a_test = useModel.test(args, model, device, test_loader, 'Testing Data')
        loss_train.append(l_train)
        acc_train.append(a_train)
        loss_test.append(l_test)
        acc_test.append(a_test)

        # If the test accuracy is the best for now, we save the model and its caracteristics
        if a_test > best_acc_test:
            best_acc_test = a_test
            best_epoch = epoch
            b_a_train = a_train
            b_l_test = l_test
            b_l_train = l_train
            torch.save(model.state_dict(), os.path.join(folder_path, all_name + '.pt'))
            print('|----\tBest test accuracy for now --> saving the model\t----|')

    print('MODEL TRAINING END, best test accuray : {0}'.format(int(best_acc_test)))

    summaryDict = {     # This dictionnary is the summary of the training
        'loss_train': loss_train,
        'acc_train': acc_train,
        'loss_test': loss_test,
        'acc_test': acc_test,
        'nb_epochs': args.epochs + bias_epoch,
        'inputs_used': args.inputs_used,
        'best_model': {     # Caracteristics of the saved model (with the best test accuracy)
            'epoch': best_epoch,
            'loss_train': b_l_train,
            'acc_train': b_a_train,
            'loss_test': b_l_test,
            'acc_test': best_acc_test
        },
        'dn_parameters': dn_parameters,
        'input_parameters': input_parameters,
        'normalization_values': normalization_values,
    }
    np.save(os.path.join(folder_path, all_name + '.npy'), summaryDict)  # Save the dictionnary
    ig.saveFigures(     # Save the plot of the evolution of the loss and accuracy for the test dans train
        folder=os.path.abspath(folder_path),
        name=all_name,
        summaryDict=summaryDict)
    ig.saveText(        # Save a text file which summarise breifly the model saved and the training
        folder=os.path.abspath(folder_path),
        name=all_name,
        summaryDict=summaryDict)
    print('Model saved in {0}'.format(folder_path))     # Show to the user where the model is saved


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
