import torch
import progressbar
import numpy as np
from torchvision import transforms

from InputGeneration.DCASEDataset import DCASEDataset


# Creates a Tensor from the Numpy dataset, which is used by the GPU for processing
class ToTensor(object):
    def __call__(self, sample):
        data, label = sample
        waveform, spectrogram, features, fmstd = data

        data_torch = (
            torch.from_numpy(waveform),
            torch.from_numpy(spectrogram),
            torch.from_numpy(features),
            torch.from_numpy(fmstd),
        )

        return data_torch, torch.from_numpy(label)


# Code for Normalization of the data
class Normalize(object):
    def __init__(
            self,
            mean_waveform, std_waveform,
            mean_spectrogram, std_spectrogram,
            mean_features, std_features,
            mean_fmstd, std_fmstd
    ):
        self.mean_waveform, self.std_waveform = mean_waveform, std_waveform
        self.mean_spectrogram, self.std_spectrogram = mean_spectrogram, std_spectrogram
        self.mean_features, self.std_features = mean_features, std_features
        self.mean_fmstd, self.std_fmstd = mean_fmstd, std_fmstd

    def __call__(self, sample):
        data, label = sample
        waveform, spectrogram, features, fmstd = data

        waveform = (waveform - self.mean_waveform) / self.std_waveform
        spectrogram = (spectrogram - self.mean_spectrogram) / self.std_spectrogram
        features = (features - self.mean_features) / self.std_features
        fmstd = (fmstd - self.mean_fmstd) / self.std_fmstd

        data = waveform, spectrogram, features, fmstd

        return data, label


def NormalizeData(train_labels_dir, root_dir, save_dir, light_data=False):
    # Load the dataset
    dcase_dataset = DCASEDataset(
        csv_file=train_labels_dir,
        root_dir=root_dir,
        save_dir=save_dir,
        light_data=light_data
    )

    # flag for the first element
    flag = 0

    # concatenate the datas computed inputs
    waveformConcat = np.asarray([])
    spectrogramConcat = np.asarray([])
    featuresConcat = np.asarray([])
    fmstdConcat = np.asarray([])

    # generate a random permutation, because it's fun. there's no specific reason for that.
    rand = np.random.permutation(len(dcase_dataset))

    # for all the training samples
    nb_files = len(dcase_dataset)
    bar = progressbar.ProgressBar(maxval=nb_files, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', progressbar.ETA()])
    bar.start()
    for i in range(nb_files):

        # extract the sample
        if light_data:
            sample = dcase_dataset[i]
        else:
            sample = dcase_dataset[rand[i]]

        data_computed, label = sample
        waveform, spectrogram, features, fmstd = data_computed
        if flag == 0:
            # get the data and init melConcat for the first time
            waveform_mean = np.mean(waveform)        # (2, 120000) -> (1,)
            waveform_mean2 = np.mean(np.square(waveform))        # (2, 120000) -> (1,)
            spectrogram_mean = np.mean(spectrogram, axis=(0, 2))     # (2, 1025, 431) -> (1025,)
            spectrogram_mean2 = np.mean(np.square(spectrogram), axis=(0, 2))     # (2, 1025, 431) -> (1025,)
            features_mean = np.mean(np.reshape(features, (5, 2, -1)), axis=(1, 2))     # (10, 431) -> (5,)
            features_mean2 = np.mean(np.reshape(np.square(features), (5, 2, -1)), axis=(1, 2))     # (10, 431) -> (5,)
            fmstd_mean = np.mean(fmstd, axis=0)     # (2, 10) -> (10,)
            fmstd_mean2 = np.mean(np.square(fmstd), axis=0)     # (2, 10) -> (10,)
            flag = 1
        else:
            # concatenate the features :
            waveform_mean += np.mean(waveform)        # (2, 120000) -> (1,)
            waveform_mean2 += np.mean(np.square(waveform))        # (2, 120000) -> (1,)
            spectrogram_mean += np.mean(spectrogram, axis=(0, 2))     # (2, 1025, 431) -> (1025,)
            spectrogram_mean2 += np.mean(np.square(spectrogram), axis=(0, 2))     # (2, 1025, 431) -> (1025,)
            features_mean += np.mean(np.reshape(features, (5, 2, -1)), axis=(1, 2))     # (10, 431) -> (5,)
            features_mean2 += np.mean(np.reshape(np.square(features), (5, 2, -1)), axis=(1, 2))     # (10, 431) -> (5,)
            fmstd_mean += np.mean(fmstd, axis=0)     # (2, 10) -> (10,)
            fmstd_mean2 += np.mean(np.square(fmstd), axis=0)     # (2, 10) -> (10,)

        bar.update(i + 1)
    bar.finish()

    waveform_mean /= nb_files
    waveform_mean2 /= nb_files
    wavform_std = waveform_mean2 - np.square(waveform_mean)

    spectrogram_mean /= nb_files
    spectrogram_mean2 /= nb_files
    spectrogram_std = spectrogram_mean2 - np.square(spectrogram_mean)

    features_mean /= nb_files
    features_mean2 /= nb_files
    features_std = features_mean2 - np.square(features_mean)

    fmstd_mean /= nb_files
    features_mean2 /= nb_files
    fmstd_std = fmstd_mean2 - np.square(fmstd_mean)

    normalization_values = {
        'waveform': (np.array([waveform_mean]), np.array([wavform_std])),
        'spectrogram': (spectrogram_mean, spectrogram_std),
        'features': (features_mean, features_std),
        'fmstd': (fmstd_mean, fmstd_std)
    }

    return normalization_values

def return_data_transform(normalization_values):
    # Load of the values in the file
    waveform_mean, waveform_std = normalization_values['waveform']  # (1,), (1,)
    spectrogram_mean, spectrogram_std = normalization_values['spectrogram']  # (1025,), (1025,)
    features_mean, features_std = normalization_values['features']  # (5,), (5,)
    fmstd_mean, fmstd_std = normalization_values['fmstd']       # (10,), (10,)

    # Create the good shape for applying operations to the tensor
    waveform_mean = np.concatenate([waveform_mean, waveform_mean])[:, np.newaxis]  # (2, 1)
    waveform_std = np.concatenate([waveform_std, waveform_std])[:, np.newaxis]  # (2, 1)
    spectrogram_mean = np.concatenate([spectrogram_mean[:, np.newaxis], spectrogram_mean[:, np.newaxis]],
                                      axis=1).T[:, :, np.newaxis]  # (2, 1025, 1)
    spectrogram_std = np.concatenate([spectrogram_std[:, np.newaxis], spectrogram_std[:, np.newaxis]],
                                     axis=1).T[:, :, np.newaxis]  # (2, 1025, 1)
    features_mean = np.reshape(  # (10, 1)
        np.concatenate(
            [
                features_mean[:, np.newaxis],
                features_mean[:, np.newaxis]
            ],
            axis=1
        ),
        (10, 1)
    )
    features_std = np.reshape(  # (10, 1)
        np.concatenate(
            [
                features_std[:, np.newaxis],
                features_std[:, np.newaxis]
            ],
            axis=1
        ),
        (10, 1)
    )
    fmstd_mean = fmstd_mean[np.newaxis, :]  # (1, 10)
    fmstd_std = fmstd_std[np.newaxis, :]    # (1, 10)

    # convert to torch variables
    waveform_mean, waveform_std = torch.from_numpy(waveform_mean), torch.from_numpy(waveform_std)
    spectrogram_mean, spectrogram_std = torch.from_numpy(spectrogram_mean), torch.from_numpy(spectrogram_std)
    features_mean, features_std = torch.from_numpy(features_mean), torch.from_numpy(features_std)
    fmstd_mean, fmstd_std = torch.from_numpy(fmstd_mean), torch.from_numpy(fmstd_std)

    # init the data_transform
    data_transform = transforms.Compose([
        ToTensor(), Normalize(
            waveform_mean, waveform_std,
            spectrogram_mean, spectrogram_std,
            features_mean, features_std,
            fmstd_mean, fmstd_std
        )
    ])

    return data_transform
