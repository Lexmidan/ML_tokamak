# Machine Learning on TOKAMAK data

## Project Overview

This Python Machine Learning project focuses on building, training, and evaluating models for classifying confinement modes of plasma in COMPASS Tokamak. The project is organized into multiple files and notebooks for various stages of the machine learning pipeline.
Models are implemented in PyTorch. Access to the data used for training and testing is available through CDBClient (internal compass module).

The core classifier architectures:

- **ResNet34** - Based on single image from RIS1/RIS2 fast camera indicates L/H/ELM confinement mode.
- **PhyDNet**[6] - Physics informed model, originally purposed for forecasting is modified to a classifier. Receives $n$ subsequent images from RIS1 and predicts the confinement mode corresponding to the last image in the sequence.
- **InceptionTime** - advanced LSTM used for time-series classification. Receives a small time-window of 1d signals (up to 4 signal sources can be chosen from 4 Mirnov Coils, Langmuire probe on divertor and $H_\alpha$) and predicts the confinement mode in arbitrary moment in the signal window.
- **Simple1DCNN** - A rather simple convolutional network, that has the same task and data as InceptionTime, but learns faster, and apparently has better results.

## Project flow

1. Prepare data using `imgs_processing.py` (resp.  `process_data_for_alt_models.py` for the 1d signal models)
2. Train and test single-image ResNet model using `confinement_mode_classifier.py`, respectively you can train and test physics informed neural network PhyDNet (see [6]) in `PhyDNet_COMPASS.py`. For 1d signal models run `alt_models_training.py`
3. Results of the training together with tensorboard events and hyperparameter will be stored in `./runs` folder.
4. Run `results_visualization.ipynb` for interactive results visualization.

## Files and Notebooks

Here is an overview of the project's directory structure and the purpose of each file:

0. `vyzkumny_ukol`: This is a folder that contains all the relevant files.

1. `imgs_processing.py`: This Python script generates datasets used for training and testing. Running this script requires CDBClient to be installed. All the data preprocess is contained in this script.

2. `LHmode_classifier.py`: a single camera (RIS1, RIS2 or both) model classifying between L-mode H-mode and ELM is trained and saved. Model's architecrute is ResNet18.

3. `ModelEnsembling.py`: in this script you will find code related to model ensembling. It combines two trained models created in `LHmode_classifier.ipynb`, to improve classification performance. Improvement is not significant, hence the script is abandoned

4. `test_model.ipynb`: testing of the trained models is conducted. It generates metric scores to evaluate performance in classifying confinement modes.

5. `TB_clustering.ipynb`: contains code related to exploratory data analysis for the project. It simply generates embeddings for the Tensorboard in order to study t-SNE, PCA and UMAP.

6. `alt_models.py`: this module contains the utility functions and classes for supplementary models e.g. InecptionTime and 1D-CNN, that operates on 1d signals from Mirnov Coils, Langmuir probes and $H_\alpha$

7. `confinement_mode_classifier.py`: in this file all the necessary functions and classes used in training and testing of RIS camera models are stored.

8. `Cross_validation.py`: a script that cross-validates the RIS and supplementary models using K-fold technique.

9. `alt_models_training.py`: training and testing routine of supplementary models.

10. `PhyDNet_COMPASS.py`: training and testing routine of PhyDNet model with "cold start" initialization.

11. `PhyDNet_finetuning.py`: finetuning and testing of pretrained in `PhyDNet_COMPASS.py` PhyDNet model.

12. `PhyDNet_models.py`: module containing the modified for the task PhyDNet model.

13. `process_data_for_alt_models.py`: a script that prepares data for training supplementary models.

14. `results_visualization.ipynb`: a notebook that invokes `visualize` function from `visual` module, that uses `ipywidgets` package to make an interactive visualization of models' results.




## Theory

[1] I. Goodfellow, Y. Bengio, and A. Courville: Deep Learning. The MIT Press, 2016.

[2] M. Zorek, et al.: Semi-supervised deep networks for plasma state identification. Plasma Phys. Control. Fusion 64 (2022) 125004.

[3] V. Weinzettl, et al.: Progress in diagnostics of the COMPASS tokamak. JINST 12 (2017) C12015.

[4] U. Losada, et al.: Observations with fast visible cameras in high power Deuterium plasma experiments in the JET ITER-like wall tokamak. Nuclear Mat. and Energy 25 (2020) 100837

[5] H. Zohm: Edge localized modes (ELMs). Plasma Phys. Control. Fusion 38 105  (1996)

[6] V. Le Guen, et al.: Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction. Computer Vision and Pattern Recognition (CVPR) (2020).

