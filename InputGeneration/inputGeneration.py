import numpy as np
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt

##### Personal importation
# import Pytorch.DenseNet.denseNetParameters as dn

# dn_parameters = dn.dn_parameters
# input_parameters = dn.input_parameters

def getStats(feature):
    return np.array([np.mean(feature, axis=(1, 2)), np.std(feature, axis=(1, 2))])

def getAllInputs(filename):
    audio, sr_ = sf.read(filename)    # sr=22050, mono=False  # (4800000, 2)
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
    #  mel_spectrogram - (2, 50, 469)
    n_mels = 100
    mel_spectrogram = np.array([librosa.feature.melspectrogram(y=left, sr=sr_, n_mels=n_mels),      # (2, 50, 469)
                                librosa.feature.melspectrogram(y=right, sr=sr_, n_mels=n_mels)])
    logmel_spectrogram = librosa.core.amplitude_to_db(mel_spectrogram)

    # getStats - (10,)
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
    fmstd = np.reshape(stats.T, (2, 10))    # (right+left, 2 * nb_features)

    ##### Create datas #####
    data = (
        waveform,  # (2, 120000)
        logmel_spectrogram,  # (2, 1025, 469), for mel : (2, 100, 469)
        features,  # (10, 469)
        fmstd  # (2, 10)
    )

    # data = (waveform, spectrogram, rms, zcr, mel_spectrogram, stats)
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


def createInputParametersFile(template, fileName, dn_parameters):
    waveform, spectrogram, features, fmstd = getAllInputs('./Dataset/train/audio/0.wav')
    template['spectrum']['nb_channels'], template['spectrum']['h'], template['spectrum']['w'] = spectrogram.shape
    template['audio']['nb_channels'], template['audio']['len'] = waveform.shape
    template['features']['nb_channels'], template['features']['len'] = features.shape
    template['fmstd']['len'] = fmstd.shape[0] * fmstd.shape[1]
    template['final']['len'] = dn_parameters['spectrum']['size_fc'] + dn_parameters['audio']['size_fc'] + \
                               dn_parameters['features']['size_fc'] + dn_parameters['fmstd']['layers_size'][-1]

    np.save(fileName, template)


def saveFigures(folder, name, summaryDict):
    loss_train = summaryDict['loss_train']
    loss_test = summaryDict['loss_test']
    acc_train = summaryDict['acc_train']
    acc_test = summaryDict['acc_test']
    nb_epochs = summaryDict['nb_epochs']

    x = np.arange(1, nb_epochs + 1)
    # Save of the loss
    plt.figure()
    plt.plot(x, loss_train, label='Training Loss')
    plt.plot(x, loss_test, label='Testing Loss')
    plt.title('Variation of the Loss through the buffers\n' + name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, 'LossFigure_' + name + '.png'))

    # Save the accuracy
    plt.figure()
    plt.plot(x, acc_train, label='Training Accuracy')
    plt.plot(x, acc_test, label='Testing Accuracy')
    plt.title('Variation of the Accuracy through the buffers\n' + name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value (%)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, 'AccuracyFigure_' + name + '.png'))


def saveText(folder, name, summaryDict):
    loss_train = summaryDict['loss_train'][-1]
    loss_test = summaryDict['loss_test'][-1]
    acc_train = summaryDict['acc_train'][-1]
    acc_test = summaryDict['acc_test'][-1]
    nb_epochs = summaryDict['nb_epochs']
    input_used = summaryDict['input_used']

    iu_txt = ''
    flag = False
    if input_used[0] == '1':
        iu_txt += 'waveform'
        flag = True
    if input_used[1] == '1':
        if flag:
            iu_txt += ', spectrogram'
        else:
            iu_txt += 'spectrogram'
            flag = True
    if input_used[2] == '1':
        if flag:
            iu_txt += ', features'
        else:
            iu_txt += 'features'
            flag = True
    if input_used[3] == '1':
        if flag:
            iu_txt += ', fmstd'
        else:
            iu_txt += 'fmstd'
            flag = True



    text = 'Summary of {5} :\n\n' \
           'Training Loss : {0}\n' \
           'Testing Loss : {1}\n' \
           'Training Accuracy : {2}\n' \
           'Testing Accuracy : {3}\n' \
           'Epochs : {4}\n\n' \
           'Inputs Used : {7}\t ({6})'\
        .format(
            loss_train, loss_test, acc_train, acc_test, nb_epochs, name, iu_txt, input_used
        )

    with open(os.path.join(folder, 'Summary_' + name + '.txt'), 'a') as f:
        f.write(text)


