import numpy as np
import librosa
import os
import soundfile as sf
import matplotlib.pyplot as plt

"""
    This file takes care of the computation of the inputs and to the saving of the results in training/testing task
"""

def getStats(feature):
    """

    :param feature: np array in 3 dimensions
    :return: the mean and the std of this array through the axis 1 and 2
    """
    return np.array([np.mean(feature, axis=(1, 2)), np.std(feature, axis=(1, 2))])

def getAllInputs(filename):
    """

    :param filename: a .wav file
    :return: the inputs (not normalized) for the neural network
    """
    audio, sr_ = sf.read(filename)      # (4800000, 2)
    left = audio[::2, 0]
    right = audio[::2, 1]

    # waveform - (2, 120000)
    waveform = audio.T[:, ::4]

    # spectrogram - (2, 1025, 469)
    spectrogram = np.abs(np.array([librosa.core.stft(left),
                                   librosa.core.stft(right)]))
    #  rms - (2, 1, 469)
    rms = np.array([librosa.feature.rms(left),          # Root Mean Square
                    librosa.feature.rms(right)])
    #  zcr - (2, 1, 469)
    zcr = np.array([librosa.feature.zero_crossing_rate(left),       # Zero Crossing Rate
                    librosa.feature.zero_crossing_rate(right)])
    #  sc - (2, 1, 469)
    sc = np.array([librosa.feature.spectral_centroid(left),         # Spectral Centroid
                   librosa.feature.spectral_centroid(right)])
    #  sr - (2, 1, 469)
    sr = np.array([librosa.feature.spectral_rolloff(left),          # Spectral Roll-of
                   librosa.feature.spectral_rolloff(right)])
    #  sfm - (2, 1, 469)
    sfm = np.array([librosa.feature.spectral_flatness(left),        # Spectral Flatness Mesure
                    librosa.feature.spectral_flatness(right)])
    #  mel_spectrogram - (2, 100, 469)
    n_mels = 100
    mel_spectrogram = np.array([librosa.feature.melspectrogram(y=left, sr=sr_, n_mels=n_mels),      # (2, 100, 469)
                                librosa.feature.melspectrogram(y=right, sr=sr_, n_mels=n_mels)])
    logmel_spectrogram = librosa.core.amplitude_to_db(mel_spectrogram)

    # getStats - (10,)
    stats = np.concatenate([getStats(rms), getStats(zcr),
                            getStats(sc), getStats(sr), getStats(sfm)])
    #### Reshape for the neural network #####
    # Waveform
    waveform = np.reshape(waveform, (2, 120000))


    # spectrogram
    spectrogram = np.reshape(spectrogram, (2, 1025, 469))       # Not used

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


#### Function to set up the environment

def setEnviromnent():
    """
        Creates the folders for the Generated Dataset
    """
    if not os.path.isdir('./GeneratedDataset'):
        os.mkdir('./GeneratedDataset')
    if not os.path.isdir('./GeneratedDataset/train'):
        os.mkdir('./GeneratedDataset/train')
    if not os.path.isdir('./GeneratedDataset/test'):
        os.mkdir('./GeneratedDataset/test')


def setLightEnviromnent():
    """
        Creates the folders for the Generated Dataset with a snall number of Data (for test on CPU)
    """
    if not os.path.isdir('./GeneratedLightDataset'):
        os.mkdir('./GeneratedLightDataset')
    if not os.path.isdir('./GeneratedLightDataset/train'):
        os.mkdir('./GeneratedLightDataset/train')
    if not os.path.isdir('./GeneratedLightDataset/test'):
        os.mkdir('./GeneratedLightDataset/test')


def createInputParametersFile(template, fileName, dn_parameters):
    """

    :param template: The template of the dictionnary input_parameters
    :param fileName: The path where we want to save it
    :param dn_parameters: the parameters of the neural network

        Creates the file "fileName" with the dictionnary input_parameters filled knowing the architecture of the
        Neural Network (known with dn_parameters)
    """
    waveform, spectrogram, features, fmstd = getAllInputs('./Dataset/train/audio/0.wav')
    template['spectrum']['nb_channels'], template['spectrum']['h'], template['spectrum']['w'] = spectrogram.shape
    template['audio']['nb_channels'], template['audio']['len'] = waveform.shape
    template['features']['nb_channels'], template['features']['len'] = features.shape
    template['fmstd']['len'] = fmstd.shape[0] * fmstd.shape[1]
    template['final']['len'] = dn_parameters['spectrum']['size_fc'] + dn_parameters['audio']['size_fc'] + \
                               dn_parameters['features']['size_fc'] + dn_parameters['fmstd']['layers_size'][-1]

    np.save(fileName, template)


def saveFigures(folder, name, summaryDict):
    """

    :param folder: the folder where we want to save it
    :param name: the name of the figures
    :param summaryDict: the data of the training we want to plot

        Save the plot of the evolution of the training loss and the testing loss through the epochs
        Save the plot of the evolution of the training accuracy and the testing loss accuracy the epochs
    """
    loss_train = summaryDict['loss_train']
    loss_test = summaryDict['loss_test']
    acc_train = summaryDict['acc_train']
    acc_test = summaryDict['acc_test']
    nb_epochs = summaryDict['nb_epochs']
    best_epoch = summaryDict['best_model']['epoch']
    best_loss_train = summaryDict['best_model']['loss_train']
    best_acc_train = summaryDict['best_model']['acc_train']
    best_loss_test = summaryDict['best_model']['loss_test']
    best_acc_test = summaryDict['best_model']['acc_test']

    min_loss = min(min(loss_train), min(loss_test))
    max_loss = max(max(loss_train), max(loss_test))
    min_acc = min(min(acc_train), min(acc_test))
    max_acc = max(max(acc_train), max(acc_test))

    x = np.arange(1, nb_epochs + 1)
    # Save of the loss
    plt.figure()
    plt.plot(x, loss_train, 'b', label='Training Loss')
    plt.plot(x, loss_test, 'r', label='Testing Loss')
    plt.title('Variation of the Loss through the buffers\n' + name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.plot([1, nb_epochs], [best_loss_train, best_loss_train], 'b--',
             label='Model training loss : {0}'.format(round(best_loss_train, 4)))
    plt.plot([1, nb_epochs], [best_loss_test, best_loss_test], 'r--',
             label='Model testing loss : {0}'.format(round(best_loss_test, 4)))
    plt.plot([best_epoch, best_epoch], [min_loss, max_loss], 'k--', label='Best Epoch : {0}'.format(best_epoch))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, 'LossFigure_' + name + '.png'))

    # Save the accuracy
    plt.figure()
    plt.plot(x, acc_train, 'b', label='Training Accuracy')
    plt.plot(x, acc_test, 'r', label='Testing Accuracy')
    plt.title('Variation of the Accuracy through the buffers\n' + name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy value (%)')
    plt.plot([1, nb_epochs], [best_acc_train, best_acc_train], 'b--',
             label='Model train accuracy : {0}'.format(round(best_acc_train, 2)))
    plt.plot([1, nb_epochs], [best_acc_test, best_acc_test], 'r--',
             label='Model test accuracy : {0}'.format(round(best_acc_test, 2)))
    plt.plot([best_epoch, best_epoch], [min_acc, max_acc], 'k--', label='Best Epoch : {0}'.format(best_epoch))
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder, 'AccuracyFigure_' + name + '.png'))


def saveText(folder, name, summaryDict):
    """

    :param folder: the folder where we want to save it
    :param name: the name of the figures
    :param summaryDict: the data of the training we want to plot

        Save a text file which summarize the saved model
    """
    loss_train = summaryDict['best_model']['loss_train']
    loss_test = summaryDict['best_model']['loss_test']
    acc_train = summaryDict['best_model']['acc_train']
    acc_test = summaryDict['best_model']['acc_test']
    nb_epochs = summaryDict['nb_epochs']
    best_epoch = summaryDict['best_model']['epoch']
    input_used = summaryDict['inputs_used']

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
           'Train Epochs : {4}\n' \
           'Best Epoch : {8}\n\n' \
           'Inputs Used : {7}\t ({6})'\
        .format(
            loss_train, loss_test, acc_train, acc_test, nb_epochs, name, iu_txt, input_used, best_epoch
        )

    with open(os.path.join(folder, 'Summary_' + name + '.txt'), 'a') as f:
        f.write(text)


