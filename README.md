# Emotion Recognition using ResNet CNN Architecture

This project attempts to recognize user emotion using a convolutional
neural network (CNN). The particular architecture used is a residual
neural network based (ResNet).

The neural net can recognize 7 emotions with relatively high accuracy:
(1) Anger, (2) Disgust, (3) Fear, (4) Happy, (5) Sad, (6) Surprise and
(7) Neutral.

The dataset for training the neural net came from the Carrier and
Courville Facial Expression Dataset hosted on Kaggle.

### How to Run:

##### Emotion Recognition

(1) In order to get going quickly, run the face_tracking.py file and
the program will begin to track your emotions via webcam.

##### Neural Net Training

(1) The neural net can be re-trained to obtain a different model via
the emotion_recognition.py file.

The current model has an accuracy of ~94.8% on the test dataset.

##### Hardware Requirements
(1) Webcam, (2) NVIDIA graphics card with ~8GB RAM

###### Note: Program has only been tested under Ubuntu 14.04


### References:
(1) https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py

(2) https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

