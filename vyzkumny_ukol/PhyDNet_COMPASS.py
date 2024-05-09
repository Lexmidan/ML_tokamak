from skimage.metrics import structural_similarity as ssim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryPrecision, BinaryRecall, F1Score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import random
import time
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall
from PhyDNet.models.models_modified import ConvLSTM,PhyCell, ClassifierRNN
from PhyDNet.data.moving_mnist import MovingMNIST
from PhyDNet.constrain_moments import K2M
import torch.multiprocessing as mp

import argparse
from tqdm import tqdm
import os
import pandas as pd
from torchvision.io import read_image
from datetime import datetime
from pathlib import Path
import confinement_mode_classifier as cmc
from torch.utils.tensorboard import SummaryWriter


class ImagesDataset(Dataset):
    '''
    takes annotations, img_dir and index, returns sequence of 10 images. label and path correspond to the last image in the sequence
    '''
    def __init__(self, annotations, img_dir, n_frames_input=10,
                 device = torch.device("cuda:0"), gray_scale = False):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.n_frames_input = n_frames_input
        self.device = device
        self.gray_scale = gray_scale
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        prev_imgs = torch.tensor([]).to(self.device) #images from the past - input for the model
        for i in range(self.n_frames_input):
            img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-4*i, 'filename'])
            image = read_image(img_path).float().to(self.device)
            image = image[:,74:-74,144:-144] # crop the image 640x500 -> 352x352
            if self.gray_scale:
                image = image.mean(dim=0, keepdim=True)

            normalized_image = (image/255)
            prev_imgs = torch.cat((prev_imgs, normalized_image.unsqueeze(0)), dim=0)

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        path = self.img_labels.iloc[idx]['filename']
  
        labeled_imgs_dict = {'imgs_input': prev_imgs,
                             'label':label, 'path':path, 'time':time}


        return labeled_imgs_dict
    
def train_on_batch(input_tensor, labels_tensor, classifier, encoder_optimizer, criterion, 
                   device = torch.device("cuda:0"), constraints = None):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    loss = 0
    for ei in range(input_length-1): 
        outputs = classifier(input_tensor[:,ei,:,:,:], (ei==0) )
        loss += criterion(outputs, labels_tensor.long() if len(labels_tensor.size())==1 else labels_tensor)


    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,classifier.phycell.cell_list[0].input_dim):
        filters = classifier.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   

    return outputs, loss/input_length


def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict,
                 writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25,
                 chkpt_path=os.getcwd(), signal_name='img', device = torch.device("cuda:0"),
                 return_best_model = True, constraints = None):

    since = time.time()

    best_acc = 0.0

    total_loss = {'train': 0.0, 'val': 0.0}
    total_batch = {'train': 0, 'val': 0}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_batch = 0

            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                inputs = batch[signal_name].to(device).float()
                labels = batch['label'].to(device)

                if len(labels.size()) > 1: #If soft labels are used
                    _, ground_truth = torch.max(labels, axis=1)
                else: 
                    ground_truth = labels

                running_batch += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, loss = train_on_batch(inputs, labels, model, optimizer, criterion, constraints=constraints, device=device)

                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        total_batch['train'] += 1
                        total_loss['train'] = (0.995*(total_loss['train']) + 0.005*loss.item())/(1-0.005**total_batch['train'])
                    else:
                        total_batch['val'] += 1
                        total_loss['val'] = (0.995*(total_loss['val']) + 0.005*loss.item())/(1-0.005**total_batch['val'])                      

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == ground_truth.data) #How many correct answers
                
                
                #tensorboard part
                
                if running_batch % int(len(dataloaders[phase])/10)==int(len(dataloaders[phase])/10)-1: 
                    # ...log the running loss
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing loss', total_loss[phase] / total_batch[phase],
                                    total_batch[phase])
                    
                    #F1 metric
                    writer.add_scalar(f'{phase}ing F1 metric',
                                    F1Score(task="multiclass", num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                    epoch * len(dataloaders[phase]) + running_batch)
                    
                    #Precision recall
                    writer.add_scalar(f'{phase}ing macro Precision', 
                                        MulticlassPrecision(num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                    writer.add_scalar(f'{phase}ing macro Recall', 
                                        MulticlassRecall(num_classes=outputs.size()[1]).to(device)(preds, ground_truth),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                writer.add_scalar(f'accuracy', epoch_acc, epoch)
                writer.close()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), chkpt_path)

        time_elapsed = time.time() - since

        # load best model weights
    if return_best_model:
        model.load_state_dict(torch.load(chkpt_path))
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model

    


def train_and_eval_PhyDNet(batch_size=8, learning_rate_min=0.0001, learning_rate_max=0.001, num_epochs=10,
                           test_run=False, test_df_contains_val_df=True):
     # data range 0 to 1 - images normalized this way
    
    timestamp = datetime.fromtimestamp(time.time()).strftime("%y-%m-%d, %H-%M-%S ")
    save_name = timestamp + ' phydnet'
    path = Path(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    #### Create dataloaders ########################################
    shot_usage = pd.read_csv(f'{path}/data/shot_usage.csv')
    shot_for_ris = shot_usage[shot_usage['used_for_ris1']]
    shot_numbers = shot_for_ris['shot']
    shots_for_testing = shot_for_ris[shot_for_ris['used_as'] == 'test']['shot']
    shots_for_validation = shot_for_ris[shot_for_ris['used_as'] == 'val']['shot']
    shots_for_training = shot_for_ris[shot_for_ris['used_as'] == 'train']['shot']

    if test_df_contains_val_df:
        shots_for_testing = pd.concat([shots_for_testing, shots_for_validation])

    if test_run:
        shots_for_testing = shots_for_testing[:3]
        shots_for_validation = shots_for_validation[:3]
        shots_for_training = shots_for_training[:3]

    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_training, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=True, ris_option='RIS1', use_for_PhyDNet=True)

    #Read article, see PhyDNet/constrain_moments.py
    constraints = torch.zeros((49,7,7)).to(device)
    ind = 0
    for i in range(0,7):
        for j in range(0,7):
            constraints[ind,i,j] = 1
            ind +=1   
    train_dset = ImagesDataset(train_df, path, gray_scale=True,n_frames_input=4)
    test_dset = ImagesDataset(test_df, path, gray_scale=True,n_frames_input=4)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=1)

    phycell  =  PhyCell(input_shape=(88,88), input_dim=352, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
    convcell =  ConvLSTM(input_shape=(88,88), input_dim=352, hidden_dims=[8,352], n_layers=2, kernel_size=(3,3), device=device)   
    classifier = ClassifierRNN(phycell, convcell, device)

    writer = SummaryWriter(f'PhyDNet/runs/{save_name}')
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate_min)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate_max, total_steps=num_epochs) #!!!
    
    model_path = Path(f'PhyDNet/runs/{save_name}/model.pt')


    trained_model = train_model(classifier, criterion, optimizer, exp_lr_scheduler, {'train':train_loader, 'val':test_loader}, writer, 
                                {'train':len(train_dset), 'val':len(test_dset)}, num_epochs=num_epochs, 
                                chkpt_path=model_path.with_name(f'{model_path.stem}_chkpt{model_path.suffix}'), 
                                signal_name='imgs_input', device=device, constraints=constraints, return_best_model=True)
    
    torch.save(trained_model.state_dict(), model_path)

    metrics = cmc.test_model(f'PhyDNet/runs/{save_name}', trained_model, test_loader,
                              comment='', writer=writer, signal_name='img', num_classes=3)
    
    metrics['prediction_df'].to_csv(f'{path}/runs/{timestamp}_all_layers/prediction_df.csv')

    metrics_per_shot = cmc.per_shot_test(path=f'PhyDNet/runs/{save_name}', 
                            shots=shots_for_testing.values.tolist(), results_df=metrics['prediction_df'],
                            writer=writer, num_classes=3,
                            two_images=False)
    
    metrics_per_shot = pd.DataFrame(metrics_per_shot)
    metrics_per_shot.to_csv(f'{path}/runs/{timestamp}_all_layers/metrics_per_shot.csv')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    train_and_eval_PhyDNet()
    print('Done')