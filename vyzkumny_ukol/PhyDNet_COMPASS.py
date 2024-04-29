from skimage.metrics import structural_similarity as ssim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import random
import time
from PhyDNet.models.models_modified import ConvLSTM,PhyCell, EncoderRNN
from PhyDNet.data.moving_mnist import MovingMNIST
from PhyDNet.constrain_moments import K2M
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
    def __init__(self, annotations, img_dir, n_frames_input=10, n_frames_output=10, 
                 device = torch.device("cuda:0"), gray_scale = False):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.device = device
        self.gray_scale = gray_scale
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        prev_imgs = torch.tensor([]).to(self.device) #images from the past - input for the model
        for i in range(self.n_frames_input):
            img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-i, 'filename'])
            image = read_image(img_path).float().to(self.device)
            image = image[:,74:-74,144:-144] # crop the image 640x500 -> 352x352
            if self.gray_scale:
                image = image.mean(dim=0, keepdim=True)

            normalized_image = (image/255)
            prev_imgs = torch.cat((prev_imgs, normalized_image.unsqueeze(0)), dim=0)
                
        futur_imgs = torch.tensor([]).to(self.device) #images from the future - target for the model
        for i in range(self.n_frames_input):
            img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-i, 'filename'])
            image = read_image(img_path).float().to(self.device)
            image = image[:,74:-74,144:-144] # crop the image 640x500 -> 352x352
            if self.gray_scale:
                image = image.mean(dim=0, keepdim=True)

            normalized_image = (image/255)
            futur_imgs = torch.cat((futur_imgs, normalized_image.unsqueeze(0)), dim=0)

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        path = self.img_labels.iloc[idx]['filename']
  
        labeled_imgs_dict = {'imgs_input': prev_imgs, 'imgs_target': futur_imgs,
                             'label':label, 'path':path, 'time':time}


        return labeled_imgs_dict
    
def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, 
                   teacher_forcing_ratio, constraints, device = torch.device("cuda:0")):                
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length  = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, output_image,_,_ = encoder(input_tensor[:,ei,:,:,:], (ei==0) )
        loss += criterion(output_image,input_tensor[:,ei+1,:,:,:])

    decoder_input = input_tensor[:,-1,:,:,:] # first decoder input = last image of input sequence
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False 
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
        target = target_tensor[:,di,:,:,:]
        loss += criterion(output_image,target)
        if use_teacher_forcing:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image

    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, nepochs, print_every=1,eval_every=1,name='', device = torch.device("cuda:0"), 
               train_loader = None, test_loader = None, batch_size = 1, data_range = 1.0, constraints = None, writer = None):
    best_ssim = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(),lr=0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2,factor=0.1,verbose=True)
    criterion = nn.MSELoss()
    loss_epoch = 0
    for epoch in range(0, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
        i=0
        for out in tqdm(train_loader, desc='train'):
            input_tensor = out['imgs_input'].to(device)
            target_tensor = out['imgs_target'].to(device)
            loss = train_on_batch(input_tensor, target_tensor, encoder, 
                                  encoder_optimizer, criterion, teacher_forcing_ratio,
                                  constraints)                                   
            loss_epoch += loss
            if i % 10 == 0:
                writer.add_scalar(f'Training MSE', loss, i)
            i+=1
         
        if (epoch+1) % eval_every == 0:

            mse, mae,ssim = evaluate(encoder,test_loader, device = device, batch_size = batch_size, data_range = data_range) 
            scheduler_enc.step(mse)                   

            writer.add_scalar(f'Eval MSE', mse, epoch)
            writer.add_scalar(f'Eval MAE', mae, epoch)
            writer.add_scalar(f'Eval SSIM', ssim, epoch)

            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(encoder.state_dict(),f'PhyDNet/{name}runs/encoder_best_ssim.pth')
    torch.save(encoder.state_dict(),f'PhyDNet/runs/{name}/encoder.pth')                           
    return encoder

    
def evaluate(encoder, loader, device = torch.device("cuda:0"), batch_size = 1, data_range = 1.0):
    total_mse, total_mae, total_ssim, total_bce = 0,0,0,0
    t0 = time.time()
    with torch.no_grad():
        i=0
        for out in tqdm(loader, desc=f'evaluation'):# mse: {total_mse/len(loader):.2f}, mae: {total_mae/len(loader):.2f}, ssim: {total_ssim/len(loader):.2f}'):

            input_tensor = out['imgs_input'].to(device)
            target_tensor = out['imgs_target'].to(device)
            input_length = input_tensor.size()[1] # num of frames in input sequence
            target_length = target_tensor.size()[1] # num of frames in target sequence

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input, False, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions =  np.stack(predictions) # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0,1)  # (batch_size,10, 1, 64, 64)

            mse_batch = np.mean((predictions-target)**2 , axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(predictions-target) ,  axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            
            for a in range(0,target.shape[0]):
                for b in range(0,target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,], data_range=data_range) / (target.shape[0]*target.shape[1]) 

            cross_entropy = -target*np.log(predictions) - (1-target) * np.log(1-predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (batch_size*target_length)
            total_bce +=  cross_entropy
            i+=1

    return total_mse/len(loader),  total_mae/len(loader), total_ssim/len(loader)
    
def train_and_eval_PhyDNet(batch_size=8,eval_every=1, print_every=1, nepochs=20, data_range = 1.0, root='PhyDNet/data/'):
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

    shot_df, test_df, val_df, train_df = cmc.load_and_split_dataframes(path,shot_numbers, shots_for_training, shots_for_testing, 
                                                                    shots_for_validation, use_ELMS=3, ris_option='RIS1', use_for_PhyDNet=True)

    #Read article, see PhyDNet/constrain_moments.py
    constraints = torch.zeros((49,7,7)).to(device)
    ind = 0
    for i in range(0,7):
        for j in range(0,7):
            constraints[ind,i,j] = 1
            ind +=1   
    train_dset = ImagesDataset(train_df, path, gray_scale=True)
    test_dset = ImagesDataset(test_df, path, gray_scale=True)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=0)

    phycell  =  PhyCell(input_shape=(88,88), input_dim=352, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
    convcell =  ConvLSTM(input_shape=(88,88), input_dim=352, hidden_dims=[8,8,352], n_layers=3, kernel_size=(3,3), device=device)   
    encoder  = EncoderRNN(phycell, convcell, device)

    writer = SummaryWriter(f'PhyDNet/runs/{save_name}')
    trained_enc = trainIters(encoder,nepochs, print_every=1, eval_every=1, 
                                name=save_name, device=device, train_loader=train_loader, 
                                test_loader=test_loader, batch_size=1, data_range=1.0, 
                                constraints=constraints, writer=writer)
