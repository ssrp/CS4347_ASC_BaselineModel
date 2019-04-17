import os
import numpy as np
import DataGeneration.inputGeneration as ig
import DataGeneration.outputGeneration as og
from torch.utils.data import Dataset

"""
    This file takes care of the loader of the data in the differents files .wav or .npy
"""

class DCASEDataset(Dataset):
    """
    Made for the training/testing task
    """
    def __init__(self, csv_file, root_dir, save_dir, transform=None, light_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            save_dir (string): The path where the computed inputs will be saved in .npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
            light_data (bool, optional): if we are working on a small number of data (test on cpu)
        """

        data_list = []
        label_list = []
        label_indices = []
        with open(csv_file, 'r') as f:
            content = f.readlines()
            content = content[2:]
            flag = 0
            for x in content:
                if flag == 0:
                    row = x.split(',')
                    data_list.append(row[0])  # first column in the csv, file names
                    label_list.append(row[1])  # second column, the labels
                    label_indices.append(row[2])  # third column, the label indices (not used in this code)
                    flag = 1
                else:
                    flag = 0
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.transform = transform
        self.datalist = data_list
        self.labels = label_list
        self.default_labels = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                               'street_pedestrian', 'street_traffic', 'tram']

        # Test if light training
        self.light_train = light_data
        if self.light_train:
            self.datalist = self.datalist[0:5]
            self.labels = self.labels[0:5]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wav_name = self.datalist[idx]
        wav_path = os.path.join(self.root_dir, wav_name)
        npy_name = os.path.splitext(
            os.path.split(wav_name)[1])[0] + '.npy'  # The name of the file with the computed inputs
        npy_path = os.path.join(
            self.save_dir,
            npy_name
        )

        data_computed = None
        if os.path.exists(npy_path):        # If it exists we load it
            data_computed = np.load(npy_path)
        else:                               # If not we create it
            data_computed = ig.getAllInputs(os.path.abspath(wav_path))
            np.save(npy_path, data_computed)

        # extract the label
        label = np.asarray(self.default_labels.index(self.labels[idx]))

        # final sample
        sample = (data_computed, label)

        # perform the transformation (normalization etc.), if required
        if self.transform:
            sample = self.transform(sample)

        return sample


class DCASEDataset_evaluation(Dataset):
    """
    Made for the evaluation task
    """
    def __init__(self, csv_file, root_dir, save_dir, transform=None, light_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the audio.
            save_dir (string): The path where the computed inputs will be saved in .npy files
            transform (callable, optional): Optional transform to be applied
                on a sample.
            light_data (bool, optional): if we are working on a small number of data (test on cpu)
        """

        data_list = []
        with open(csv_file, 'r') as f:
            content = f.readlines()
            content = content[2:]
            flag = 0
            for x in content:
                if flag == 0:
                    data_list.append(x[:-1])  # first column in the csv, file names
                    flag = 1
                else:
                    flag = 0
        self.save_dir = save_dir
        self.root_dir = root_dir
        self.transform = transform
        self.datalist = data_list
        self.default_labels = og.labels


        # Test if light training
        self.light_data = light_data
        if self.light_data:
            self.datalist = self.datalist[0:5]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        wav_name = self.datalist[idx]
        wav_path = os.path.join(self.root_dir, wav_name)
        npy_name = os.path.splitext(os.path.split(wav_name)[1])[0] + '.npy'
        npy_path = os.path.join(
            self.save_dir,
            npy_name
        )

        if os.path.exists(npy_path):
            data_computed = np.load(npy_path)
        else:
            data_computed = ig.getAllInputs(os.path.abspath(wav_path))
            np.save(npy_path, data_computed)

        # perform the transformation (normalization etc.), if required
        if self.transform:
            data_computed = self.transform(data_computed)

        return data_computed, idx       # idx will serve to keep the order of the data
