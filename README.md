# :one: CS4347_ASC_BaselineModel
This repo is dedicated to CS4347 Group Project on Acoustic Scene Classification. The goal of acoustic scene classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded.

## Dataset :page_facing_up:

You can access the training dataset on [this Google Drive Link](https://drive.google.com/drive/u/1/folders/1HaMgbk2Heszdj71b_6H20-J01Xh8M3u8). It is already divided into train and test sets. You can start downloading the dataset as it is about 10GB+.

The dataset for this project is the TUT Urban Acoustic Scenes 2018 dataset, consisting of recordings from various acoustic scenes. The dataset was recorded in six large european cities, in different locations for each scene class. For each recording location there are 5-6 minutes of audio. The original recordings were split into segments with a length of 10 seconds that are provided in individual files. Available information about the recordings include the following: acoustic scene class (label).

### Acoustic Scenes / Labels
Acoustic scenes for the task (10):

- Airport - airport
- Indoor shopping mall - shopping_mall
- Metro station - metro_station
- Pedestrian street - street_pedestrian
- Public square - public_square
- Street with medium level of traffic - street_traffic
- Travelling by a tram - tram
- Travelling by a bus - bus
- Travelling by an underground metro - metro
- Urban park - park

## GPU Access
Once you download the dataset, you would need to set up your GPU Access. Please let us know if you need help with the GPU access (SoC NUS provides Sunfire cluster for this).

## Baseline Model

The baseline system provides a simple entry-level state-of-the-art approach that gives reasonable results. The baseline system is built on librosa toolbox written in PyTorch.

Students are strongly encouraged to build their own systems by extending the provided baseline system. The system has all needed functionality for the dataset handling, acoustic feature storing and accessing, acoustic model training and storing, and evaluation. The modular structure of the system enables participants to modify the system to their needs. The baseline system is a good starting point especially for the entry level students interested in Deep Learning using PyTorch to familiarize themselves with the acoustic scene classification problem.

### Model Structure

The baseline system implements a convolutional neural network (CNN) based approach, where log mel-band energies are first extracted for each 10-second signal, and a network consisting of two CNN layers and one fully connected layer is trained to assign scene labels to the audio signals. Given below is the model structure:

<p align = "center">
<img src="images/baseline.png" width="600">
</p>

#### Parameters
- Acoustic features
  - Analysis frame 40 ms (50% hop size)
  - Log mel-band energies (40 bands)
- Network Structure
  - Input shape: 40 * 500 (10 seconds)
  - Architecture:
    - CNN layer #1
      - 2D Convolutional layer (filters: 32, kernel size: 7) + Batch normalization + ReLu activation
      - 2D max pooling (pool size: (5, 5)) + Dropout (rate: 30%)
    - CNN layer #2
      - 2D Convolutional layer (filters: 64, kernel size: 7) + Batch normalization + ReLu activation
      - 2D max pooling (pool size: (4, 100)) + Dropout (rate: 30%)
    - Flatten
    - Dense layer #1
      - Dense layer (units: 100, activation: ReLu )
      - Dropout (rate: 30%)
    - Output layer (activation: softmax)
  - Learning (epochs: 200, batch size: 16, data shuffling between epochs)
    - Optimizer: Adam (learning rate: 0.001)
  - Model selection:
    - Model performance after each epoch is evaluated on the test set, and best performing model is selected.

### Baseline Results :chart_with_downwards_trend:
To be announced soon. (56.8%)

---

# :two: Our personal model

For this project we created 2 files : ```modelTrain.py``` and ```modelEvaluate.py```.
- ```modelTrain.py``` is used to create/load and train a model.
- ```modelEvaluate.py``` is used to load a model and use it to classify .wav files.

## The architecture of the neural network

### The inputs

The neural network takes 4 inputs :
- 1) The waveform of the audio (2 channels : right and left)
- 2) The mel spectrogram (2 channels : right and left)
- 3) An input called "features" : We first divide the the audios in buffers of length 2048 and 0.5 overlap. Then we extract 5 features on each buffers, the result is the evolution of these features through the time in the audio file
  - The Root Mean Square (RMS)
  - The Zero Crossing Rate (ZCR)
  - The Spectral Centroid (SC)
  - The Spectral Roll-Off (SRO)
  - The Spectral Flatness Mesure (SFM)
- 4) The mean and the standard deviation of these features through all the buffers of the audio file (input called "fmsdt" : "features-mean-standard-deviation)

### DenseNet Model

The DenseNet model as been introduced by Gao Huang et al. as an ameloration of the ResNet architecture. (Paper : https://arxiv.org/pdf/1608.06993.pdf)
It is composed with several blocks as followed (in a block, there is a skip connection next to each convolutional layers and a concatenation layer taking as input the skip connection and the outpout of the convolutional layer

<p align = "center">
<img src="images/DenseNet.PNG" width="375" height="300">
</p>

Between each "DenseNet Block" There is a transition block with a 2x2 Max Pooling :

<p align = "center">
<img src="images/DenseNetTransition.PNG" width=150" height="100">
</p>

### Our DenseNetPerso

Our Model is based on the DenseNet architecture as shown in the following figure :

<p align = "center">
<img src="images/Model Architecture.png" width=600" height="300">
</p>
                                                                
It is possible to choose between different model (with different parameters). The model with the best result has the parameters :
```python
{
        'name': 'big',
        'spectrum': {
            'k': 32,  # The number of channel in the denseNet
            'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
            'size_fc': 100  # Size of the fully connected at the end
        },
        'audio': {
            'k': 32,  # The number of channel in the denseNet
            'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
            'size_fc': 100  # Size of the fully connected at the end
        },
        'features': {
            'k': 32,  # The number of channel in the denseNet
            'nb_blocks': 4,  # The number of dense block in the NN = len(nb_conv)
            'nb_conv': [4, 4, 4, 4],  # The numbers of convolutional layers in a dense block
            'size_fc': 50,  # Size of the fully connected at the end
        },
        'fmstd': {
            'nb_layers': 3,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [80, 50, 20],  # The size of the layers in fully connected layers
        },
        'final': {  # The parameters for the fully connected layers at the end of the neural network
            'nb_layers': 3,  # The number of fully connected layers in the NN = len(layers_size)
            'layers_size': [80, 40, 10],  # The size of the layers in the fully connected layers
        }
    },      # Big
```

## The architecture of the project :open_file_folder:

The project structure with the usable code is this one : 

```shell
.CS4347_ASC_GroupProject
|-- .DataGeneration   # This folder contains all the file for the input and output processing
    |-- DCASEDataset.py   # Contains le loaders (classes which give the inputs to the neural network)
    |-- dataNormalization.py  # Take care of the normalization of the inputs before they go in the neural network
    |-- inputGeneration.py    # Functions to extract the waveform/spectrogram/features/fmstd from the .wav files.
    |-- outputGeneration.py   # Functions to save the outputs of the neural network when we use it to make predictions
|-- .Pytorch      # Folder with all the neural network in it
    |-- .DenseNet   # This folder contains all the file to create a DenseNetPerso Model
        |-- DenseNetPerso.py    # Class DenseNetPerso : create the architecture described before
        |-- denseNetParameters.py     # The parameters of the differents models
    |-- useModel.py     # Functions train/test/evaluate to use the neural network created
|-- modelTrain.py   # The code to create/load a model and train it
|-- modelEvaluate.py    # The code to load a model and use it to make predictions

```

This is the structure of all the data saved or used : (all this folder are ignored in the git repository)

```shell
.CS4347_ASC_GroupProject
|-- .Dataset*   # This folder contains the dataset to the train an test part
    |-- .train
        |-- .audio    # All the .wav file for the train part
            |-- 0.wav
                ..
        |-- train_labels.csv
    |-- .test
        |-- .audio    # All the .wav file for the test part
            |-- 4286.wav
                ..
        |-- test_labels.csv
|-- .evaluation_data*      # The dataset for the evaluation part
    |-- .audio
        |-- 0.wav
            ..
    |-- evaluate_labels
|-- .GeneratedDataset**     The saved files of the extracted inputs of the .wav files in Dataset
    |-- .train
        |-- 0.npy
        ..
     |-- .test
        |-- 4286.npy
        ..
     |-- input_parameters.npy     # Caracteristic of the inputs for the neural network
     |-- normalization_values.npy   # Values to normalize the data before going in the neural network
|-- .GeneratedEvaluateDataset**     The saved files of the extracted inputs of the .wav files in evaluation_data
    |-- 0.npy
        ..
|-- .GeneratedLightDataset**     The saved files of the extracted inputs of the .wav files in Dataset (for working with a small number of data)
    |-- .train
        |-- 0.npy
        ..
     |-- .test
        |-- 4286.npy
        ..
     |-- input_parameters.npy     # Caracteristic of the inputs for the neural network
     |-- normalization_values.npy   # Values to normalize the data before going in the neural network
|-- .GeneratedLightEvaluateDataset**     The saved files of the extracted inputs of the .wav files in evaluation_data (for working with a small number of data)
    |-- 0.npy
        ..
|-- .SavedEvaluations**    # Contains the saved outputs of the models for the evaluation task
    |-- .big               # id of the model
        |-- .Model(big)_InputsUsed(1111)_NbEpochs(200)_(0)   # The name of the model saved
            |-- predict_labels.csv    # The .csv file with the prediction of the labels asked
            |-- predictionsDict.npy   # The prediction of the labels in the .npy file
        ..
    ..
|-- .SavedModels**         # Contains all the needed information of the model to load it and its training
    |-- .big               # id of the model
        |-- .Model(big)_InputsUsed(1111)_NbEpochs(200)_(0)   # The name of the model saved
            |-- AccuracyFigure_Model(big)_InputsUsed(1111)_NbEpochs(200)_(0).png    # The plot of the evolution of the testing and training accuracy through the epochs
            |-- LossFigure_Model(big)_InputsUsed(1111)_NbEpochs(200)_(0).png    # The plot of the evolution of the testing and training loss through the epochs
            |-- Model(big)_InputsUsed(1111)_NbEpochs(200)_(0).npy   # All the information needed to load the model and the informations about its training
            |-- Model(big)_InputsUsed(1111)_NbEpochs(200)_(0).npt   # The saved model
            |-- Summary_Model(big)_InputsUsed(1111)_NbEpochs(200)_(0).txt   # Text file which summarize briefly the model saved
        ..
    ..

```

```folder*``` means that the folder has to be added to the project by hand

```folder**``` means that the folder is automatically created by the project

## How to use ```modelTrain.py``` :clock130:

### The commands line :computer:

To train a model, the command line is : (ou have to be in the folder CS4347_ASC_GroupProject)
```shell
python modelTrain.py
```

### The Description of the Arguments :clipboard:

You can add multiple arguments :
#### Arguments from the baselign
- ```--batch-size=16``` : The size of the batches for the training
- ```--test-batch-size=16``` : The size of the batches for the testing
- ```--epochs=50```: the number of epochs for the train part
- ```--lr=0.001```: the value of the learning rate for the training part
- ```--no-cuda=False```: disable CUDA training
- ```--seed=1```: random seed
- ```--log-interval=20```: how many batch to wait before logging training status
- ```--no-saved-model=False```: For saving the model at the end of the training
#### Personal arguments

- ```--name=''```: If we want to give a name to our model (will appear in the name of the folder where it is saved)
- ```--model-id=big```: The id of our model : the dn_parameters['name'] in the file. It can be :
  - ```small```
  - ```medium```
  - ```mediumTest```
  - ```big```
- ```--inputs-used=1111```: Booleans saying if the input is used : ```1``` means "used" and ```0``` means "not used". And the order is : waveform, spectrum, features, fmstd. For example ```1111``` means that we want to use all the inputs. `1001` means that we want to use only the waveform and the fmstd.
- ```--load-model=''``` : if we want to load a model. There is 2 cases :
  - 1) The model has no name and then it is stored in a folder like this : "Model(```big```)_InputUsed(```1111```)_NbEpochs(```300```)_(```0```)". Then to load this model, you have to write :
  ```shell
  python modelTrain.py --load-model big-1111-300-0
  ```
  - 2) The model has name and then it is stored in a folder like this : "Model(```big```)_InputUsed(```1111```)_(```myName```)_NbEpochs(```300```)_(```0```)". Then to load this model, you have to write :
  ```shell
  python modelTrain.py --load-model big-1111-myName-300-0
  ```
 
##### To test if it works on CPU
- ```--light-train=False```: Set to True if you want to train on a small number of data (to test on CPU for example)
- ```--light-test=False```: Set to True if you want to test on a small number of data (to test on CPU for example)
- ```--light-data=False```: If True, set ```--light-train``` and ```--light-test``` to True
- ```--light-all=False```: If True, set ```--light-data``` to true and ```--model-id``` to 'small'

### What it does :bulb:

If asked, it will load the model ```--load-model```.
It will train the model ```--model-id``` through ```--epochs``` epochs.
And save everything in the folder :
"./SavedModel/```--model-id```/Model(```--model-id```)\_(```--inputs-used```)\_NbEpochs(```-epochs```)\_(```id```)"
if it has no name, if it has a name it will save everything in the folder :
"./SavedModel/```--model-id```/Model(```--model-id```)\_(```--inputs-used```)\_(```myName```)\_NbEpochs(```-epochs```)\_(```id```)".
```id``` is just a number created by the code to differentiate several saving of the same training.

## How to use ```modelEvaluate.py``` :heavy_check_mark:

### The commands line :computer:

To train a model, the command line is : (ou have to be in the folder CS4347_ASC_GroupProject)
```shell
python modelEvalutation.py --folder-id big-1111-300-0
```

### The Description of the Arguments :clipboard:

You can add multiple arguments :
#### Arguments from the baselign
- ```--batch-size=16``` : The size of the batches for the testing
- ```--no-cuda=False```: disable CUDA training
- ```--log-interval=10```: how many batch to wait before logging evaluating status
- ```--no-saved-model=False```: For saving the model at the end of the training
#### Personal arguments

- ```--name=''```: If we want to give a name to our evaluation
- ```--folder-name=''``` is the name of the folder where everything of the model we want to load is stored. For example :
  ```shell
  python modelEvaluate.py --folder-name Model(big)_InputUsed(1111)_NbEpochs(300)_(0)
  ```
- ```--folder-id=''``` : is the name of the folder where everything of the model we want to load is stored. The id of the folder is construct has followed :
  ```shell
  --folder-id big-1111-300-0
  ```
  will do the save action as the option
  ```shell
  --folder-name Model(big)_InputUsed(1111)_NbEpochs(300)_(0)
  ```
  or the option
  ```shell
  --folder-id big-1111-myName-300-0
  ```
  will do the save action as the option
  ```shell
  --folder-name Model(big)_InputUsed(1111)_(myName)_NbEpochs(300)_(0)
  ```
  If ```--folder-name``` and ```--folder-id``` is specified, ```--folder-id``` win and we don't look at the argument ```--folder-name```

 
##### To test if it works on CPU
- ```--light-test=False```: Set to True if you want to test on a small number of data (to test on CPU for example)


### What it does :bulb:

It will load the model ```--load-model```.
It will evaluate the data in the folder "evaluation_data" and save everything in the folder :
"./SavedEvaluation/```id of the model```/```name of the folder loaded```"
It will save in it the prediction in the file ```predicted_labels.csv``` and ```predictionsDict.npy```
:warning:
If the folder already exists, it delete the old files and creates the new ones with the new predictions.
:warning:


## Our Results :chart_with_upwards_trend:
We managed to get a test accuracy of ***82%*** 
