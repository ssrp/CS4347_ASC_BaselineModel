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
from DataGeneration import inputGeneration as ig
import DataGeneration.dataNormalization as dn
import DataGeneration.outputGeneration as og
from DataGeneration.DCASEDataset import DCASEDataset_evaluation
from Pytorch.DenseNet.DenseNetPerso import DenseNetPerso
import Pytorch.DenseNet.denseNetParameters as DN_param
import Pytorch.useModel as useModel


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Can be use to do a evaluation for the ASC project (CS4347)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # Personal arguments
    parser.add_argument('--light-test', action='store_true', default=False,
                        help='For testing on a small number of data')
    parser.add_argument('--light-all', action='store_true', default=False,
                        help='--light-test & small model')
    parser.add_argument('--name', default='',
                        help='The name of the model')
    parser.add_argument('--folder-name', default='',
                        help='The name of the folder with the saved model')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = 'evaluation_data'
    labels_dir = 'evaluation_data/evaluate_labels.csv'


    ##### Creation of the folders for the Generated Dataset #####

    # The arguments 'light' are made to test the network on my poor little computer and its poor little CPU
    if args.light_all:
        args.model_id = 'small'
    light_train = args.light_all or args.light_data or args.light_train  # Only 5 files in the train dataset
    light_test = args.light_all or args.light_data or args.light_test  # Only 5 files in the test dataset

    model_name = args.folder_name
    model_folder = 'SavedModels/{0}/{1}'.format(model_name.split(')')[0][6:], model_name)

    summaryDict = np.load(os.path.join(model_folder, model_name + '.npy')).item()

    dn_parameters = summaryDict['dn_parameters']
    input_parameters = summaryDict['input_parameters']
    normalization_values = summaryDict['normalization_values']
    inputs_used = summaryDict['inputs_used']

    if light_train:
        # If we want to test on a little CPU
        g_data_dir = './GeneratedLightEvaluateDataset/'
        if not os.path.isdir(g_data_dir):
            os.mkdir(g_data_dir)
    else:
        # Made for training on GPU (big dataset size)
        g_data_dir = './GeneratedEvaluateDataset/'
        if not os.path.isdir(g_data_dir):
            os.mkdir(g_data_dir)

    # Transformers to normalize the inputs at the entrance of the neural network
    data_transform = dn.return_data_transform(normalization_values, evaluation=True)


    # init the datasets
    dcase_dataset = DCASEDataset_evaluation(
        csv_file=labels_dir,
        root_dir=data_dir,
        save_dir=g_data_dir,
        transform=data_transform,
        light_data=light_test)

    # set number of cpu workers in parallel
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

    # Get the training and testing data loader
    evaluate_loader = torch.utils.data.DataLoader(
        dcase_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)

    # init the model
    model = DenseNetPerso(
        dn_parameters=dn_parameters,
        input_parameters=input_parameters,
        inputs_used=inputs_used
    ).to(device)
    # Load the trained model
    model.load_state_dict(torch.load(os.path.join(model_folder, model_name + '.pt')))


    # Create the architecture of the saved predictions
    if not os.path.isdir('./SavedEvaluations'):      # One big folder "SavedModels"
        os.mkdir('./SavedEvaluations')
    if not os.path.isdir(os.path.join('./SavedEvaluations', model_name.split(')')[0][6:])):
        os.mkdir(os.path.join('./SavedEvaluations', model_name.split(')')[0][6:]))

    saved_evaluations_folder = os.path.join('./SavedEvaluations', model_name.split(')')[0][6:], model_name)
    if not os.path.isdir(saved_evaluations_folder):
        os.mkdir(saved_evaluations_folder)

    predictions, indexes = useModel.evaluate(args, model, device, evaluate_loader)   # The predicted index
    print('predictions : {0}'.format(predictions))
    predictions_label = og.return_predicted_labels(predictions)             # The predicted labels
    predictionsDict = {
        'index': predictions,
        'labels': predictions_label,
        'indexes': indexes
    }
    print('indexes : {0}'.format(indexes))
    np.save(os.path.join(saved_evaluations_folder, 'predictionsDict.npy'), predictionsDict)      # Save the dict

    # Save the .csv file
    og.create_csv(
        template_path=labels_dir,
        save_path=os.path.abspath(os.path.join(saved_evaluations_folder, 'predicted_labels.csv')),
        predictions_label=predictions_label,
        predictions=predictions,
        indexes=indexes
    )

if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
