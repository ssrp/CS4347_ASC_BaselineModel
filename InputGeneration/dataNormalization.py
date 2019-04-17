import torch
import progressbar
import numpy as np

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


def NormalizeData(dcase_dataset, light_data=False):
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
