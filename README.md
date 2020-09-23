# Deep Learning based Visual Odometry
> Visual Odometry using a Recurrent Convolutional Neural Network in PyTorch

This is a PyTorch implementation of the paper `DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks`
[link](https://arxiv.org/abs/1709.08429). 

![alt text](arch.png)


## Problem Statement

Predict the current pose of the vehicle based on the previous poses, from a sequence of camera images, using an end-to-end deep learning approach

## Solution
Use a recurrent convolutional neural network with the CNN part used to extract useful features from the images and the LSTM to perform sequential modeling. Thus this method
- bypasses complete geometric pipeline (no camera calibration required as well)
- takes as input just sequence of RGB images

## Network architecture
> 9-layer CNN followed by a 2-layer LSTM. CNN architecture inspired from FlowNet [link](https://arxiv.org/abs/1504.06852)

CNN Details:
- There are 9 Conv layers with Leaky ReLU activation function
- Dropout and batch normalization employed to avoid overfitting 
- Size of the kernels gradually reduced from 7x7 to 3x3
- Number of channels increase from 64 to 1024.
- Output of last Conv layer is 20*6*1024 dimensional vector 

LSTM Details:
- 2 Layer LSTM each with a hidden unit size of 1000
- Each layer has between 5-7 repeating units

![alt text](conv_lstm.png)

## Requirements

The codebase is implemented in Python 3.7
To install the necessary requirements, run the following commands:

If you use the python shipped virtual environments:
```
python3 -m venv <your_env_name>
source your_env_name/bin/activate
pip3 install -r requirements.txt
```

If you use conda:
```
conda create <your_env_name>
conda activate your_env_name 
conda install --yes --file requirements.txt
while read requirement; do conda install --yes $requirement; done < requirements.txt
```

## Dataset
The network is trained and tested on the `KITTI Vision Benchmark Suite`[link](http://www.cvlibs.net/datasets/kitti/), a very popular dataset used for odometry and SLAM. Sequences `00, 01, 02, 05, 09` were used for training and sequences `04, 06, 07, 10` were used for inference

> Download pretrained model from : https://drive.google.com/open?id=1FfBokYsSSfMGV-FeTskNHAYaKXAZWAx_

## Data Preprocessing
The following data pre-processing was performe on the data:

- The input to the network is a sequence of monocular images taken from a camera on-board from KITTI Vision Benchmark Suite
- The color images were downsampled to 608*184 in order to suit the computational resources.
- The image is then normalized by subtracting the mean and dividing by the standard deviation. 
- The image sequence was sub-sequenced into smaller sequences of 5-7 images. 
- For each subsequence, two consecutive images were stacked to form a tensor of dimension (608,184,6)
- Each such tensor is an input to Conv1 layer of the CNN.
- Number of timesteps for the RNN was randomly chosen between 5 and 7 since larger timesteps exhausted the GPU resources. 
- The dataset contains a 12 dimensional vector as groundtruth pose for each image. 
- Internally, it consists of the rotation matrix (9) and the translation vector (3). 
- The rotation matrix is converted to Euler angles using Rodrigues' equation and the translation vector is used as is. The 6 dimensional output is saved in a numpy file.
- The output of the network is a 6 dimensional vector - 3 for Euler angles indicating orientation and 3 for x,y,z indicating translation.


## Parameters

Some of the important parameters used for the network are as follows

```
  --img_means            (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
  --img_stds             (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
  --img_size             (608,184)
  --rnn_hidden_size      1000
  --epochs               250
  --optim                Adagrad
  --learning_rate        0.0005
```

## Examples

_Training the model_

```
python main.py 
```

_Testing using trained model_

```
python test.py 
```

## Contact

Akshay Iyer – [@akshay_iyerr](https://twitter.com/akshay_iyerr) – akshay.iyerr@gmail.com

[Portfolio](https://akshay-iyer.github.io/)

