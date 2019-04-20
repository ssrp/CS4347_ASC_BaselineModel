from __future__ import print_function, division

import argparse
import os
# Ignore warnings
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Personal imports
import DataGeneration.dataNormalization as dn
import DataGeneration.outputGeneration as og
from DataGeneration.DCASEDataset import DCASEDataset_evaluation
from Pytorch.DenseNet.DenseNetPerso import DenseNetPerso
import Pytorch.useModel as useModel

"""
    This file is use to make predictions with a model
"""


def main():
    """
        main function which will do everything to create predictions
    """
    # Training settings
    parser = argparse.ArgumentParser(description='Can be use to do a evaluation for the ASC project (CS4347)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging evaluating status')
    # Personal arguments
    parser.add_argument('--light-test', action='store_true', default=False,
                        # If we want to work on a small number of testing data (for test on CPU)
                        help='For testing on a small number of data')
    parser.add_argument('--name', default='',  # Optional, if we want ot name our model
                        help='The name of the evaluation')
    parser.add_argument('--folder-name', default='',  # If we want to load a model in a folder
                        help='The name of the folder with the saved model')
    parser.add_argument('--folder-id', default='',
                        # if we want to load a model without writting all the long folder name
                        help='modelName-inputsUsed(-name)-nbEpochs-idx')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = 'evaluation_data'
    labels_dir = 'evaluation_data/evaluate_labels.csv'

    ##### Creation of the folders for the Generated Dataset #####

    # The arguments 'light' are made to test the network on my poor little computer and its poor little CPU
    light_test = args.light_test  # Only 5 files in the test dataset

    if args.folder_id != '':  # folder-id win on the name of the model
        folder_id = args.folder_id.split('-')
        model_id = folder_id[0]
        hasName = len(folder_id) == 5
        nom = '_Name({0})'.format(folder_id[2]) if hasName else ''
        b = 1 if hasName else 0
        model_name = 'Model({0})_InputsUsed({1}){2}_NbEpochs({3})_({4})'.format(  # Reconstruction of the folder name
            model_id,           # small, big, medium
            folder_id[1],       # inputs used
            nom,                # name (optional)
            folder_id[2+b],     # nb epochs
            folder_id[3+b]      # index
        )
        model_folder = 'SavedModels/{0}/{1}'.format(folder_id[0], model_name)
    else:
        model_name = args.folder_name
        model_id = model_name.split(')')[0][6:]
        model_folder = 'SavedModels/{0}/{1}'.format(model_id, model_name)

    summaryDict = np.load(
        os.path.join(model_folder, model_name + '.npy')
    ).item()        # The dictionnary with qll the informations of the model

    dn_parameters = summaryDict['dn_parameters']        # Parameters of the DenseNet Model
    input_parameters = summaryDict['input_parameters']      # Informations on the shape of the inputs
    normalization_values = summaryDict['normalization_values']      # The values used to normalize the inputs
    inputs_used = summaryDict['inputs_used']        # The inputs used by our model

    if light_test:
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
    if not os.path.isdir(os.path.join('./SavedEvaluations', model_id)):
        os.mkdir(os.path.join('./SavedEvaluations', model_id))

    saved_evaluations_folder = os.path.join('./SavedEvaluations', model_id, model_name)
    if not os.path.isdir(saved_evaluations_folder):     # The folder where we will save everything
        os.mkdir(saved_evaluations_folder)

    predictions, indexes = useModel.evaluate(args, model, device, evaluate_loader)   # The predicted index
    predictions_label = og.return_predicted_labels(predictions)             # The predicted labels
    predictions_dict = {
        'index': predictions,
        'labels': predictions_label,
        'indexes': indexes
    }
    # Save the prediction_dict dictionary
    np.save(os.path.join(saved_evaluations_folder, 'predictionsDict.npy'), predictions_dict)      # Save the dict

    # Save the .csv file
    og.create_csv(
        template_path=labels_dir,
        save_path=os.path.abspath(os.path.join(saved_evaluations_folder, 'predicted_labels.csv')),
        predictions_label=predictions_label,
        predictions=predictions,
        indexes=indexes
    )
    print('Predictions saved in {0}'.format(saved_evaluations_folder))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
