import os
import re
import time 
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from torch import cuda
import torchvision
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pytorch_lightning as pl

import confinement_mode_classifier as cmc


def cross_validation():
    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pl.seed_everything(42)
    #Where are csv spreadsheets with time, mode, img_path, h_alpha columns are saved
    data_dir_path = f'{path}/data/LH_alpha'
    file_names = os.listdir(data_dir_path)
    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_ris1']]
    shot_numbers = shot_for_ris['shot']

    #Shuffle the shot numbers
    shot_numbers = shot_numbers.sample(frac=1)
    shot_numbers.reset_index(drop=True, inplace=True)

    #Create a dataframe with 5 folds
    k_fold_dataframe = pd.DataFrame({f'{i}_fold': ['train']*len(shot_numbers) for i in range(1,6)}, index=shot_numbers.values)
    k_fold_dataframe.index.name = 'shot_number'
    for i in range(5):
        k_fold_dataframe.loc[shot_numbers[9*i:9*i+9], f'{i+1}_fold'] = 'val'
    #Save the k fold shot distribution into csv
    k_fold_dataframe.to_csv(f'{path}/cross_validation_of_RIS1_model/k_fold.csv')

    #Train and test the model for each fold
    for fold in k_fold_dataframe.columns:
        shots_for_train = k_fold_dataframe[k_fold_dataframe[fold] == 'train'].index
        shots_for_validation = k_fold_dataframe[k_fold_dataframe[fold] == 'val'].index
        shots_for_testing = shots_for_validation

        train_and_test_ris_model(comment_for_model_name=f'{fold}', shot_numbers=shot_numbers, 
                                shots_for_testing=shots_for_testing, shots_for_validation=shots_for_validation)



def train_and_test_ris_model(ris_option = 'RIS1',
                            num_workers = 32,
                            num_epochs_for_fc = 10,
                            num_epochs_for_all_layers = 10,
                            num_classes = 2,
                            batch_size = 32,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = 'ris1',
                            shot_numbers=None,
                            shots_for_testing=None,
                            shots_for_validation=None):
    
    """
    Trains a one ris model. The model is trained on RIS1 images or RIS2 images.
    """

    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    #### Create dataloaders ########################################
    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=False)
    
    test_dataloader = cmc.get_dloader(test_df, path, batch_size, 
                                        ris_option=ris_option, balance_data=False, 
                                        shuffle=False, num_workers=num_workers)

    val_dataloader = cmc.get_dloader(val_df, path, batch_size, 
                                        ris_option=ris_option, balance_data=True, 
                                        shuffle=False, num_workers=num_workers)

    train_dataloader = cmc.get_dloader(train_df, path, batch_size, 
                                        ris_option=ris_option, balance_data=True, 
                                        shuffle=False, num_workers=num_workers)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    #### Create model and train it #################################
    timestamp =  comment_for_model_name
    writer = SummaryWriter(f'cross_validation_of_RIS1_model/last_fc/{timestamp}')

    # Load a pretrained model and reset final fully connected layer.
    pretrained_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes) #3 classes: L-mode, H-mode, ELM
    pretrained_model = pretrained_model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=learning_rate_min)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=num_epochs_for_fc)

    #### Train the last fully connected layer########################
    model_path = Path(f'{path}/cross_validation_of_RIS1_model/last_fc/{timestamp}/model.pt')
    model = cmc.train_model(pretrained_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_fc, 
                        chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))


    #### Train the whole model######################################
    writer = SummaryWriter(f'cross_validation_of_RIS1_model/all_layers/{timestamp}')
    for param in model.parameters():
        param.requires_grad = True

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_min)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=num_epochs_for_all_layers)

    #Path to save the model
    model_path = Path(f'{path}/cross_validation_of_RIS1_model/all_layers/{timestamp}/model.pt')


    model = cmc.train_model(model, criterion, optimizer, exp_lr_scheduler, 
                            dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_all_layers,
                            chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))
    torch.save(model.state_dict(), model_path)

    #### Test the model############################################
    metrics = cmc.test_model(f'cross_validation_of_RIS1_model/all_layers/{timestamp}', model, test_dataloader, comment='', writer=writer)
    img_path = cmc.per_shot_test(path=f'{path}/cross_validation_of_RIS1_model/all_layers/{timestamp}/', 
                                shots=shots_for_testing, results_df=metrics['prediction_df'], writer=writer)
    writer.add_scalar(f'Accuracy on test_dataset', metrics['accuracy'])
    writer.add_scalar(f'F1 metric on test_dataset', metrics['f1'])
    writer.add_scalar(f'Precision on test_dataset', metrics['precision'])
    writer.add_scalar(f'Recall on test_dataset', metrics['recall'])
    writer.close()

    return model, model_path

if __name__ == "__main__":
    cross_validation()