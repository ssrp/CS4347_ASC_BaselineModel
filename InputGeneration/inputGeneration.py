import numpy as np
import librosa
import os

##### Personal importation
# import Pytorch.DenseNet.denseNetParameters as dn

# dn_parameters = dn.dn_parameters
# input_parameters = dn.input_parameters

def getStats(feature):
    return np.array([[np.mean(feature[0], axis=1), np.std(feature[0], axis=1)],
                     [np.mean(feature[1], axis=1), np.std(feature[1], axis=1)]])

def getAllInputs(filename):
    audio, _ = librosa.core.load(filename, sr=22050, mono=False)
    left = audio[0]
    right = audio[1]

    if os.path.isfile(os.path.splitext(filename)[0] + ".npy"):
        print("Generating input from .npy!")
        data = np.load(os.path.splitext(filename)[0] + ".npy")
        return data
    else:
        print("Generating input from audio file!")
        # waveform - (2, 220500, 1)
        waveform = np.array([[[i] for i in left], 
                             [[i] for i in right]])
        # spectrogram - (2, 1025, 431)
        spectrogram = np.abs(np.array([librosa.core.stft(left),
                                       librosa.core.stft(right)]))
        #  rms - (2, 1, 431)
        rms = np.array([librosa.feature.rms(left), 
                        librosa.feature.rms(right)])
        #  zcr - (2, 1, 431)
        zcr = np.array([librosa.feature.zero_crossing_rate(left), 
                        librosa.feature.zero_crossing_rate(right)])
        #  sc - (2, 1, 431)
        sc = np.array([librosa.feature.spectral_centroid(left),
                       librosa.feature.spectral_centroid(right)])
        #  sr - (2, 1, 431)
        sr = np.array([librosa.feature.spectral_rolloff(left),
                       librosa.feature.spectral_rolloff(right)])
        #  sfm - (2, 1, 431)
        sfm = np.array([librosa.feature.spectral_flatness(left),
                        librosa.feature.spectral_flatness(right)])
        #  mel_spectrogram - (2, 128, 431)
        mel_spectrogram = np.array([librosa.feature.melspectrogram(left), 
                                    librosa.feature.melspectrogram(right)])
        # getStats - (5, 2, 1)
        stats = np.concatenate([getStats(rms), getStats(zcr), 
                                getStats(sc), getStats(sr), getStats(sfm)])


        #### Reshape for the neural network #####
        # Waveform
        waveform = np.reshape(waveform, (2, 220500))

        # spectrogram
        # Good

        # Features
        features = np.concatenate(
            [
                np.reshape(rms, (2, 431)),
                np.reshape(zcr, (2, 431)),
                np.reshape(sc, (2, 431)),
                np.reshape(sr, (2, 431)),
                np.reshape(sfm, (2, 431))
            ],
            axis=0
        )

        # Features mstd
        fmstd = np.reshape(stats, (1, 10))

        ##### Create datas #####
        data = (
            waveform,   # (2, 220500)
            spectrogram,    # (2, 1025, 431)
            features,   # (10, 431)
            fmstd       # (1, 10)
        )

        #data = (waveform, spectrogram, rms, zcr, mel_spectrogram, stats)
        np.save(os.path.splitext(filename)[0] + ".npy", data)
        return data

def getFilesInput(n):
    for i in range(n):
        getAllInputs(f"../Dataset/train/audio/{i}.wav")
