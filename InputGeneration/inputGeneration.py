import numpy as np
import librosa
import os
import soundfile as sf
import pickle

##### Personal importation
# import Pytorch.DenseNet.denseNetParameters as dn

# dn_parameters = dn.dn_parameters
# input_parameters = dn.input_parameters

def getStats(feature):
    return np.array([np.mean(feature, axis=(1, 2)), np.std(feature, axis=(1, 2))])

def getAllInputs(filename):
    audio, _ = sf.read(filename)    # sr=22050, mono=False  # (4800000, 2)
    left = audio[::2, 0]
    right = audio[::2, 1]

    # waveform - (2, 120000)
    waveform = audio.T[:, ::4]

    # spectrogram - (2, 1025, 469)
    spectrogram = np.abs(np.array([librosa.core.stft(left),
                                   librosa.core.stft(right)]))
    #  rms - (2, 1, 469)
    rms = np.array([librosa.feature.rms(left),
                    librosa.feature.rms(right)])
    #  zcr - (2, 1, 469)
    zcr = np.array([librosa.feature.zero_crossing_rate(left),
                    librosa.feature.zero_crossing_rate(right)])
    #  sc - (2, 1, 469)
    sc = np.array([librosa.feature.spectral_centroid(left),
                   librosa.feature.spectral_centroid(right)])
    #  sr - (2, 1, 469)
    sr = np.array([librosa.feature.spectral_rolloff(left),
                   librosa.feature.spectral_rolloff(right)])
    #  sfm - (2, 1, 469)
    sfm = np.array([librosa.feature.spectral_flatness(left),
                    librosa.feature.spectral_flatness(right)])
    #  mel_spectrogram - (2, 128, 431)
    """
    Is not used
    mel_spectrogram = np.array([librosa.feature.melspectrogram(left),
                                librosa.feature.melspectrogram(right)])
    """
    # getStats - (20,)
    stats = np.concatenate([getStats(rms), getStats(zcr),
                            getStats(sc), getStats(sr), getStats(sfm)])

    #### Reshape for the neural network #####
    # Waveform
    waveform = np.reshape(waveform, (2, 120000))


    # spectrogram
    spectrogram = np.reshape(spectrogram, (2, 1025, 469))

    # Features
    features = np.concatenate(
        [
            np.reshape(rms, (2, 469)),
            np.reshape(zcr, (2, 469)),
            np.reshape(sc, (2, 469)),
            np.reshape(sr, (2, 469)),
            np.reshape(sfm, (2, 469))
        ],
        axis=0
    )
    features = np.reshape(features, (10, 469))

    # Features mstd
    fmstd = np.reshape(stats, (2, 10))

    ##### Create datas #####
    data = (
        waveform,  # (2, 120000)
        spectrogram,  # (2, 1025, 431)
        features,  # (10, 431)
        fmstd  # (2, 10)
    )

    #data = (waveform, spectrogram, rms, zcr, mel_spectrogram, stats)
    return data

def getFilesInput(n):
    for i in range(n):
        getAllInputs(f"../Dataset/train/audio/{i}.wav")



#### Function to set up the environment

def setEnviromnent():
    if not os.path.isdir('./GeneratedDataset'):
        os.mkdir('./GeneratedDataset')
    if not os.path.isdir('./GeneratedDataset/train'):
        os.mkdir('./GeneratedDataset/train')
    if not os.path.isdir('./GeneratedDataset/test'):
        os.mkdir('./GeneratedDataset/test')


def setLightEnviromnent():
    if not os.path.isdir('./GeneratedLightDataset'):
        os.mkdir('./GeneratedLightDataset')
    if not os.path.isdir('./GeneratedLightDataset/train'):
        os.mkdir('./GeneratedLightDataset/train')
    if not os.path.isdir('./GeneratedLightDataset/test'):
        os.mkdir('./GeneratedLightDataset/test')


def returnInputParameters(template, fileName, dn_parameters):
    waveform, spectrogram, features, fmstd = getAllInputs('./Dataset/train/audio/0.wav')
    template['spectrum']['nb_channels'], template['spectrum']['h'], template['spectrum']['w'] = spectrogram.shape
    template['audio']['nb_channels'], template['audio']['len'] = waveform.shape
    template['features']['nb_channels'], template['features']['len'] = features.shape
    template['fmstd']['len'] = fmstd.shape[0] * fmstd.shape[1]
    template['final']['len'] = dn_parameters['spectrum']['size_fc'] + dn_parameters['audio']['size_fc'] + \
                               dn_parameters['features']['size_fc'] + dn_parameters['fmstd']['layers_size'][-1]



    with open(fileName, 'wb') as dump_file:
        pickle.dump(template, dump_file)

