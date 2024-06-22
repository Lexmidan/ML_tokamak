# Machine Learning on TOKAMAK data

## Project Overview

This Python Machine Learning project focuses on building, training, and evaluating models for classifying confinement modes of plasma in COMPASS Tokamak. The project is organized into multiple files and notebooks for various stages of the machine learning pipeline.
Models are implemented in PyTorch. Access to the data used for training and testing is available through CDBClient (internal compass module).
The core model architecture is ResNet18, a simple 1-D CNN might be used as a supplementary.

## Files and Notebooks

Here is an overview of the project's directory structure and the purpose of each file:

1. `imgs_processing.py`: This Python script generates datasets used for training and testing. Running this script requires CDBClient to be installed. All the data preprocess is contained in this script.


2. `LHmode_classifier.ipynb`: a single camera (RIS1 or RIS2) model classifying between L-mode H-mode and ELM is trained and saved. Model's architecrute is ResNet18.

3. `ModelEnsembling.ipynb`: in this notebook you will find code related to model ensembling. It combines two trained models created in `LHmode_classifier.ipynb`, to improve classification performance. 

4. `test_model.ipynb`: testing of the trained models is conducted. It generates metric scores to evaluate performance in classifying confinement modes.

5. `TB_clustering.ipynb`: contains code related to exploratory data analysis for the project. It simply generates embeddings for the Tensorboard in order to study t-SNE, PCA and UMAP.

6. `CNN_on_halpha.ipynb`: is a work in progress for creating a 1-D Convolutional Neural Network (CNN) model for analyzing H-alpha diagnostics. The purpose is to create a supplementary model to enhance classification of ELMs and H-modes.

7. `confinement_mode_classifier.py`: in this file all the necessary functions and classes used in training and testing are stored.

## Project Flow

Here's a rough outline of how the project flow works:

1. Data Preprocessing and Dataset Preparation: run the script `imgs_processing.py` and specify numbers of shots. Default diagnostics variant is 'seidl_2023'.

2. Model Training: `LHmode_classifier.ipynb`, imports ResNet18 model and trains the last fully connected layer with all the other layers frozen. Then the training continues with all the layers unfrozen.

3. Model Ensembling: `ModelEnsembling.ipynb` combines multiple trained models from `LHmode_classifier.ipynb`, then trains MLP only, and then procceeds a complete training of all the layers. There are two options to combine the models. First is to combine models trained on RIS1 and RIS2 camera respectively, or to combine two RIS1 models with that the second model processes images from precending time (model sees two images and classify the confinement mode in time t based on images from time t and t-dt).


4. Testing and Evaluation: `test_model.ipynb`



## Usage and Instructions

To work with this project, follow these general steps:

1. Execute the notebooks in the order mentioned above, starting with data preprocessing and ending with testing and evaluation.

2. Ensure that you have the necessary dependencies installed (mainly CDBClient).

## Theory

[1] I. Goodfellow, Y. Bengio, and A. Courville: Deep Learning. The MIT Press, 2016.

[2] M. Zorek, et al.: Semi-supervised deep networks for plasma state identification. Plasma Phys. Control. Fusion 64 (2022) 125004.

[3] V. Weinzettl, et al.: Progress in diagnostics of the COMPASS tokamak. JINST 12 (2017) C12015.

[4] U. Losada, et al.: Observations with fast visible cameras in high power Deuterium plasma experiments in the JET ITER-like wall tokamak. Nuclear Mat. and Energy 25 (2020) 100837

[5] H. Zohm: Edge localized modes (ELMs). Plasma Phys. Control. Fusion 38 105  (1996)


