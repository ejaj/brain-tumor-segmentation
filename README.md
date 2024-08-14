# Brain Tumor Segmentation using 3D UNet

This project involves training a 3D UNet model for the task of brain tumor segmentation using volumetric MRI data. The goal is to accurately segment tumor regions from 3D MRI scans.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

Brain tumor segmentation is a critical task in medical imaging that aids in the diagnosis and treatment planning for patients. This project implements a 3D UNet model, which is commonly used for medical image segmentation tasks due to its ability to capture both local and global contextual information.

## Dataset

The dataset used in this project is collected from [Synapse](https://www.synapse.org/), a collaborative platform for sharing data, tools, and ideas. The dataset consists of volumetric MRI scans stored in `.h5` files. Each file contains a 3D image of the brain and the corresponding segmentation mask, where different labels represent different tumor regions.

### Data Preprocessing

The images and masks are normalized, resized, and augmented during training to improve the model's robustness. The preprocessing steps include:
- **Normalization**: Scaling the pixel values to a standard range.
- **Resizing**: Adjusting the images and masks to a uniform size.
- **Augmentation**: Applying random transformations like flips and rotations to the data.

## Model Architecture

The model used is a 3D UNet, which consists of the following components:
- **Encoder**: A series of convolutional layers that downsample the input and extract features.
- **Bottleneck**: The central part of the network that captures the most abstract features.
- **Decoder**: A series of upsampling layers that reconstruct the segmented image from the bottleneck features.
- **Skip Connections**: Connections between the encoder and decoder layers to retain spatial information.

## Training

The model is trained using a combined loss function that includes Dice Loss and Binary Cross-Entropy (BCE) Loss. The training process includes:
- **Optimizer**: Adam optimizer with an initial learning rate of `0.001`.
- **Learning Rate Scheduler**: Reduces the learning rate when the validation loss plateaus.
- **Early Stopping**: Stops training when the validation loss does not improve for a specified number of epochs.

### Commands

To train the model, run:
```bash
python main.py
