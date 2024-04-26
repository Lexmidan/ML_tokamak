import os
import re
import time 
from pathlib import Path
import json
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

def train_and_test_ris_model(ris_option = 'RIS1',
                            pretrained_model = torchvision.models.resnet18(pretrained=True),
                            num_workers = 32,
                            num_epochs_for_fc = 10,
                            num_epochs_for_all_layers = 10,
                            num_classes = 3,
                            batch_size = 32,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = f', 3 output classes',
                            random_seed = 42,
                            augmentation = False):
    
    """
    Trains a one ris model. The model is trained on RIS1 images or RIS2 images.
    """

    comment_for_model_name = ris_option + comment_for_model_name
    pl.seed_everything(random_seed)

    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    #### Create dataloaders ########################################
    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_ris1'] if ris_option == 'RIS1' else shot_usage['used_for_ris2']]
    shot_numbers = shot_for_ris['shot']
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']
    shots_for_training = shot_for_ris[shot_for_ris['used_as'] == 'train']['shot']

    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_training, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=num_classes==3, ris_option=ris_option)


    test_dataloader = cmc.get_dloader(test_df, path, batch_size, balance_data=False, 
                                      shuffle=False, num_workers=num_workers, 
                                      augmentation=False)

    val_dataloader = cmc.get_dloader(val_df, path, batch_size, balance_data=True, 
                                     shuffle=False, num_workers=num_workers, 
                                     augmentation=False)

    train_dataloader = cmc.get_dloader(train_df, path, batch_size, balance_data=True, 
                                       shuffle=False, num_workers=num_workers, 
                                       augmentation=augmentation)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    #### Create model and train it #################################
    timestamp =  datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name
    writer = SummaryWriter(f'runs/{timestamp}_last_fc')

    # Load a pretrained model and reset final fully connected layer.
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes) #3 classes: L-mode, H-mode, ELM
    pretrained_model = pretrained_model.to(device)

    # Let's visualize the model
    # sample_input = next(iter(train_dataloader))['img']
    # print('Adding graph to tensorboard')
    # writer.add_graph(pretrained_model, sample_input.float().to(device))

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=learning_rate_min)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=num_epochs_for_fc) #!!!

    # Model will be saved to this folder along with metrics and tensorboard scalars
    model_path = Path(f'{path}/runs/{timestamp}_last_fc/model.pt')

    #### Train the last fully connected layer########################
    model = cmc.train_model(pretrained_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_fc, 
                        chkpt_path=model_path.with_name(f'{model_path.stem}_best_val_acc{model_path.suffix}'))

    hyperparameters = {
    'model': model.__class__.__name__,
    'batch_size': batch_size,
    'num_epochs': num_epochs_for_fc,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': torch.tensor(shots_for_testing.values.tolist()),
    'shots_for_validation': torch.tensor(shots_for_validation.values.tolist()),
    'shots_for_training': torch.tensor(shots_for_training.values.tolist()),
    'ris_option': ris_option,
    'num_classes': num_classes,
    'second_image': 'None',
    'augmentation': "applied" if augmentation else "no augmentation",
    'random_seed': random_seed
    }
    
    
    torch.save(model.state_dict(), model_path)

    #### Test the model############################################
    metrics = cmc.test_model(f'runs/{timestamp}_last_fc', model, test_dataloader, comment='', 
                             writer=writer, num_classes=num_classes, signal_name='img')

    img_path = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_last_fc/', 
                                shots=shots_for_testing.values.tolist(), 
                                results_df=metrics['prediction_df'], 
                                writer=writer,
                                num_classes=num_classes)

    one_digit_metrics = {'Accuracy on test_dataset': metrics['accuracy'], 
                        'F1 metric on test_dataset':metrics['f1'].tolist(), 
                        'Precision on test_dataset':metrics['precision'].tolist(), 
                        'Recall on test_dataset':metrics['recall'].tolist()}

    writer.add_hparams(hyperparameters, one_digit_metrics)
    writer.close()
    
    # Save hyperparameters and metrics to a JSON file
    for key in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
        hyperparameters[key] = hyperparameters[key].tolist()  # Convert tensors to lists
    all_hparams = {**hyperparameters, **one_digit_metrics}
    # Convert to JSON
    json_str = json.dumps(all_hparams, indent=4)
    with open(f'{path}/runs/{timestamp}_last_fc/hparams.json', 'w') as f:
        f.write(json_str)

    #### Train the whole model######################################
    torch.cuda.empty_cache()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=learning_rate_min)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=num_epochs_for_all_layers) #!!!

    writer = SummaryWriter(f'runs/{timestamp}_all_layers')
    for param in model.parameters():
        param.requires_grad = True

    model_path = Path(f'{path}/runs/{timestamp}_all_layers/model.pt')

    model = cmc.train_model(model, criterion, optimizer, exp_lr_scheduler, 
                            dataloaders, writer, dataset_sizes, num_epochs=num_epochs_for_all_layers,
                            chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'))
    
    torch.save(model.state_dict(), model_path)

    hyperparameters = {
    'batch_size': batch_size,
    'num_epochs': num_epochs_for_all_layers,
    'optimizer': optimizer.__class__.__name__,
    'criterion': criterion.__class__.__name__,
    'learning_rate_max': learning_rate_max,
    'scheduler': exp_lr_scheduler.__class__.__name__,
    'shots_for_testing': torch.tensor(shots_for_testing.values.tolist()),
    'shots_for_validation': torch.tensor(shots_for_validation.values.tolist()),
    'shots_for_training': torch.tensor(shots_for_training.values.tolist()),
    'ris_option': ris_option,
    'num_classes': num_classes,
    'second_image': 'None',
    'augmentation': "applied" if augmentation else "no augmentation",
    'random_seed': random_seed
    }

    #### Test the model############################################
    metrics = cmc.test_model(f'runs/{timestamp}_all_layers', model, test_dataloader,
                              comment='', writer=writer, signal_name='img', num_classes=num_classes)

    img_path = cmc.per_shot_test(path=f'{path}/runs/{timestamp}_all_layers/', 
                                shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
                                writer=writer, num_classes=num_classes)
    
    one_digit_metrics = {'Accuracy on test_dataset': metrics['accuracy'], 
                        'F1 metric on test_dataset':metrics['f1'].tolist(), 
                        'Precision on test_dataset':metrics['precision'].tolist(), 
                        'Recall on test_dataset':metrics['recall'].tolist()}

    writer.add_hparams(hyperparameters, one_digit_metrics)
    writer.close()
    
    # Save hyperparameters and metrics to a JSON file
    for key in ['shots_for_testing', 'shots_for_validation', 'shots_for_training']:
        hyperparameters[key] = hyperparameters[key].tolist()  # Convert tensors to lists
    all_hparams = {**hyperparameters, **one_digit_metrics}
    # Convert to JSON
    json_str = json.dumps(all_hparams, indent=4)
    with open(f'{path}/runs/{timestamp}_all_layers/hparams.json', 'w') as f:
        f.write(json_str)

    return model, model_path

if __name__ == '__main__':
    train_and_test_ris_model()
    print('Done')