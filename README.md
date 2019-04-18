# CS4347_ASC_BaselineModel
This repo is dedicated to CS4347 Group Project on Acoustic Scene Classification. The goal of acoustic scene classification is to classify a test recording into one of the provided predefined classes that characterizes the environment in which it was recorded.

## Dataset

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

### Baseline Results
To be announced soon. (56.8%)

---

# Our personnal model

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

The DenseNet model as been introduced by Gao Huang et al. as an ameloration of the ResNet architecture.
It is composed with several blocks as followed (in a block, there is a skip connection next to each convolutional layers and a concatenation layer taking as input the skip connection and the outpout of the convolutional layer

<p align = "center">
<img src="images/DenseNet.PNG" width="375" height="300">
</p>

Between each "DenseNet Block" There is a transition block with a 2x2 Max Pooling :

<p align = "center">
<img src="images/DenseNetTransition.PNG" width=150" height="100">
</p>

### Our DenseNetPerso

## The architecture of the project

## How to use ```modelTrain.py```

### The commands line

### The Description of the Arguments

## How to use ```modelEvaluate.py```

### The commands line

### The Description of the Arguments
