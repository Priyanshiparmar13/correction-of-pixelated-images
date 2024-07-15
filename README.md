# Correction-of-pixelated-images
# Detect and Correct PixelNet Images
This project focuses on detecting pixelated images using Python and its libraries, and correcting them using a DnCNN (Denoising Convolutional Neural Network) model. Additionally, a custom data generator is implemented to facilitate the training and testing of the model.

# Introduction
Pixelation is a common issue in digital images, where images are rendered at a lower resolution than intended, resulting in visible pixels. This project aims to detect such pixelated images and correct them to a higher quality using a DnCNN model.

# Features
•	Detection: Identify pixelated images using Python and image processing libraries.

•	Correction: Use the DnCNN model to improve the quality of pixelated images.

•	Custom Data Generator: Generate training and testing datasets to train the DnCNN model effectively.

# Requirements
•	Python 3.7+

•	numpy

•	tensorflow

•	keras

•	opencv-python

•	matplotlib

•	scikit-image

# Model Details
The DnCNN model is a convolutional neural network designed for image denoising tasks. It is used in this project to correct pixelated images by learning to predict the high-quality version of the input images.

Custom Data Generator

The custom data generator is designed to efficiently load and preprocess image data for training the DnCNN model. It can handle large datasets by generating data in batches and applying necessary transformations on-the-fly.

# Results
The project demonstrates significant improvement in the quality of pixelated images. 

