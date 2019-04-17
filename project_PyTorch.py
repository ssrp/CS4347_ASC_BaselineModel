from __future__ import print_function, division

import argparse
import os
# Ignore warnings
import warnings
import progressbar

import numpy as np
import torch

warnings.filterwarnings("ignore")

# import PyTorch Functionalities
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# import Librosa, tool for extracting features from audio data

# Personal imports
from InputGeneration import inputGeneration as ig
import InputGeneration.dataNormalization as dn
from InputGeneration.DCASEDataset import DCASEDataset
from Pytorch.DenseNet.DenseNetPerso import DenseNetPerso
import Pytorch.DenseNet.denseNetParameters as DN_param
import Pytorch.useModel as useModel


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Baseline code for ASC Group Project (CS4347)')
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

    parser.add_argument('--light-train', action='store_true', default=False,
                        help='For training on a small number of data')
    parser.add_argument('--light-test', action='store_true', default=False,
                        help='For testing on a small number of data')
    parser.add_argument('--light-data', action='store_true', default=False,
                        help='--light-train & --light-test')
    parser.add_argument('--light-all', action='store_true', default=False,
                        help='--light-data & small model')
    parser.add_argument('--name', default='',
                        help='The name of the model')
    parser.add_argument('--model-id', default='medium',
                        help='Model ID')
    parser.add_argument('--inputs-used', default='1111',
                        help='The inputs used (1=used, 0=not used) (waveform, spectrum, features, fmstd)')


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

    if args.light_all:
        args.model_id = 'small'

    light_train = args.light_all or args.light_data or args.light_train
    light_test = args.light_all or args.light_data or args.light_test

    dn_parameters = DN_param.return_model_parameters(args.model_id)

    if light_train:
        # If we want to test on CPU
        ig.setLightEnviromnent()
        g_train_data_dir = './GeneratedLightDataset/train/'
        g_test_data_dir = './GeneratedLightDataset/test/'
        g_data_dir = './GeneratedLightDataset/'
    else:
        ig.setEnviromnent()
        g_train_data_dir = './GeneratedDataset/train/'
        g_test_data_dir = './GeneratedDataset/test/'
        g_data_dir = './GeneratedDataset/'

    if os.path.isfile(
            os.path.join(g_data_dir, 'normalization_values.npy')
    ) and os.path.isfile(
        os.path.join(g_data_dir, 'input_parameters.npy')
    ):
        # get the mean and std. If Normalized already, just load the npy files and comment
        #  the NormalizeData() function above
        normalization_values = np.load(os.path.join(g_data_dir, 'normalization_values.npy'))
        normalization_values = normalization_values.item()      # We have to do this to access the dictionary
        input_parameters = np.load(os.path.join(g_data_dir, 'input_parameters.npy')).item()
        print(
            'LOAD OF THE FILE normalization_values.npy FOR NORMALIZATION AND input_parameters.npy FOR THE NEURAL NETWORK'
        )
    else:
        # If not, run the normalization and save the mean/std
        print('DATA NORMALIZATION : ACCUMULATING THE DATA')
        # load the dataset
        normalization_values = dn.NormalizeData(
            train_labels_dir=os.path.abspath(train_labels_dir),
            root_dir=os.path.abspath(train_data_dir),
            save_dir=os.path.abspath(g_train_data_dir),
            light_data=light_train
        )
        np.save(os.path.join(g_data_dir, 'normalization_values.npy'), normalization_values)
        ig.createInputParametersFile(
            template=DN_param.input_parameters,
            fileName=os.path.abspath(os.path.join(g_data_dir, 'input_parameters.npy')),
            dn_parameters=dn_parameters
        )
        input_parameters = np.load(os.path.join(g_data_dir, 'input_parameters.npy')).item()

        print('DATA NORMALIZATION COMPLETED')

    data_transform = dn.return_data_transform(normalization_values)

    # init the datasets
    dcase_dataset = DCASEDataset(
        csv_file=train_labels_dir,
        root_dir=train_data_dir,
        save_dir=g_train_data_dir,
        transform=data_transform,
        light_data=light_train
    )
    dcase_dataset_test = DCASEDataset(
        csv_file=test_labels_dir,
        root_dir=test_data_dir,
        save_dir=g_test_data_dir,
        transform=data_transform,
        light_data=light_test
    )

    # set number of cpu workers in parallel
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}


    # get the training and testing data loader
    train_loader = torch.utils.data.DataLoader(
        dcase_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dcase_dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs
    )

    # init the model
    model = DenseNetPerso(
        dn_parameters=dn_parameters,
        input_parameters=input_parameters,
        inputs_used=args.inputs_used
    ).to(device)

    # init the optimizer
    optimizer_adam = optim.Adam(model.parameters(), lr=args.lr)

    print('MODEL TRAINING START')

    # Prepare the training
    loss_train, acc_train, loss_test, acc_test = [], [], [], []

    if not args.no_save_model:
        if not os.path.isdir('./SavedModels'):
            os.mkdir('./SavedModels')
        model_name = DN_param.id2name(args.model_id)
        model_folder = os.path.join('./SavedModels', model_name)
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        # Find a name for the save
        if args.name != '':
            nom = '_Name({0})'.format(args.name)
        else:
            nom = ''

        flag = True
        i = 0
        while flag:
            all_name = 'Model({0})_InputsUsed({4}){1}_NbEpochs({2})_({3})'.format(
                model_name,
                nom,
                args.epochs,
                i,
                args.inputs_used
            )
            folder_path = os.path.join(model_folder, all_name)
            if not os.path.isdir(folder_path):
                flag = False
            i += 1
        os.mkdir(folder_path)

    best_acc_test = 0
    b_a_train = 0
    b_l_test = 0
    b_l_train = 0
    best_epoch = 0
    # train the model
    for epoch in range(1, args.epochs + 1):
        useModel.train(args, model, device, train_loader, optimizer_adam, epoch)
        l_train, a_train = useModel.test(args, model, device, train_loader, 'Training Data')
        l_test, a_test = useModel.test(args, model, device, test_loader, 'Testing Data')
        loss_train.append(l_train)
        acc_train.append(a_train)
        loss_test.append(l_test)
        acc_test.append(a_test)
        torch.save(model.state_dict(), os.path.join(folder_path, all_name + '.pt'))

        if a_test > best_acc_test:
            best_acc_test = a_test
            best_epoch = epoch
            b_a_train = a_train
            b_l_test = l_test
            b_l_train = l_train
            print('\t\tBest test accuracy for now --> saving the model')
    print('MODEL TRAINING END')

    summaryDict = {
        'loss_train': loss_train,
        'acc_train': acc_train,
        'loss_test': loss_test,
        'acc_test': acc_test,
        'nb_epochs': args.epochs,
        'input_used': args.inputs_used,
        'best_model': {
            'epoch': best_epoch,
            'loss_train': b_l_train,
            'acc_train': b_a_train,
            'loss_test': b_l_test,
            'acc_test': best_acc_test
        }
    }
    np.save(os.path.join(folder_path, all_name + '.npy'), summaryDict)
    ig.saveFigures(
        folder=os.path.abspath(folder_path),
        name=all_name,
        summaryDict=summaryDict
    )
    ig.saveText(
        folder=os.path.abspath(folder_path),
        name=all_name,
        summaryDict=summaryDict
    )
    print('Model saved in {0}'.format(folder_path))





if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
