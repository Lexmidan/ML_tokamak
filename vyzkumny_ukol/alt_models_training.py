import os
from pathlib import Path
import re
import time 
from datetime import datetime

import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pytorch_lightning as pl

import alt_models as am



def train_and_test_alt_model(signal_name = 'divlp',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            batch_size = 512,
                            num_workers = 8,
                            num_epochs = 12,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = '',
                            random_seed = 42):
    """ 
    Trains and tests alternative model on given signal.
    """
    pl.seed_everything(random_seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    path = Path(os.getcwd())
    comment_for_model_name = architecture + '_on_' + signal_name  + str(signal_window) +'dpoints ' + comment_for_model_name
    
    # Load data
    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_alt']]
    shot_numbers = shot_for_ris['shot']
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']

    shot_df, test_df, val_df, train_df = am.split_df(path, shot_numbers, shots_for_testing, shots_for_validation, use_ELMS=True)


    # Create dataloaders
    train_dataloader = am.get_dloader(train_df, batch_size=batch_size, 
                               balance_data=True, shuffle=False, 
                               signal_window=signal_window,
                               signal_name=signal_name,
                               num_workers=num_workers)

    val_dataloader = am.get_dloader(val_df, batch_size=batch_size, 
                                balance_data=True, shuffle=False, 
                                signal_window=signal_window,
                                signal_name=signal_name,
                                num_workers=num_workers)

    test_dataloader = am.get_dloader(test_df, batch_size=batch_size, 
                                balance_data=False, shuffle=False, 
                                signal_window=signal_window,
                                signal_name=signal_name,
                                num_workers=num_workers)

    dataloaders = {'train':train_dataloader, 'val':val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # Create tensorboard writer
    timestamp =  datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ") + comment_for_model_name
    writer = SummaryWriter(f'runs/{timestamp}')
    model_path = Path(f'{path}/runs/{timestamp}/model.pt')

    # Create model
    untrained_model = am.select_model_architecture(architecture=architecture, window=signal_window, num_classes=3)
    untrained_model = untrained_model.to(device)

    # Write model graph to tensorboard
    sample_input = next(iter(train_dataloader))[str(signal_name)].to(device).float()
    writer.add_graph(untrained_model, sample_input)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(untrained_model.parameters(), lr=learning_rate_min)

    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=50) #!!!

    # Train model
    model = am.train_model(untrained_model, criterion, optimizer, exp_lr_scheduler, 
                        dataloaders, writer, dataset_sizes, num_epochs=num_epochs, 
                        chkpt_path = model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'),
                        signal_name=signal_name)

    torch.save(model.state_dict(), model_path)

    # Test model
    metrics = am.test_model(f'{path}/runs/{timestamp}', model, test_dataloader, comment ='3 classes', signal_name=signal_name, writer=writer)

    am.per_shot_test(f'{path}/runs/{timestamp}', shots_for_testing, metrics[0], writer=writer)


if __name__ == "__main__":
    train_and_test_alt_model(signal_name = 'divlp',
                            architecture = 'InceptionTime',
                            signal_window = 320,
                            batch_size = 512,
                            num_workers = 8,
                            num_epochs = 12,
                            learning_rate_min = 0.001,
                            learning_rate_max = 0.01,
                            comment_for_model_name = '3 classes')
    print('Done')