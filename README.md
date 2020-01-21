# DeepVO_PyTorch
Visual Odometry using a Recurrent Convolutional Neural Network in PyTorch

Create a virtual env and install the required dependencies 
pip3 install -r requirements.txt

Download the KITTI dataset of image sequences for Visual Odometry

Download pretrained model from : https://drive.google.com/open?id=1FfBokYsSSfMGV-FeTskNHAYaKXAZWAx_

To infer using trained model, run:
python test.py

To train the model, run:
python main.py
