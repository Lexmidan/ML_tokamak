import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import re
import torch
import pandas as pd
import torchvision
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall, MulticlassPrecisionRecallCurve, MulticlassROC
from torchmetrics.classification import BinaryPrecision, BinaryRecall, F1Score, ConfusionMatrix, BinaryPrecisionRecallCurve, BinaryROC
from torch.optim import lr_scheduler
import torch.nn as nn
import copy
from torch.utils.tensorboard import SummaryWriter
import time 





# # Setting the seed
# pl.seed_everything(42)
# # Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print("Device:", device)



####################### Create datasets and dataloaders #######################

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, annotations, img_dir, mean, std):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mean = mean
        self.std = std


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
        image = read_image(img_path).float()
        normalized_image = (image - self.mean[:, None, None])/(255 * self.std[:, None, None])
        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        return {'img': normalized_image,'label': label, 'path': img_path, 'time': time}


class HalphaDataset(Dataset):
    '''
        Parameters:
            annotations (DataFrame): The DataFrame containing annotations for the dataset.
            window (int): The size of the time window for fetching sequential data.

    '''

    def __init__(self, annotations, h_alpha_window, img_dir):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.h_alpha_window = h_alpha_window
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
        h_window = torch.tensor([])
        
        if idx-self.h_alpha_window < 0: 
            h_window = torch.tensor(self.h_alpha_window*[self.img_labels.iloc[idx]['h_alpha']])
        else:
            h_window = torch.tensor(self.img_labels.iloc[idx-self.h_alpha_window:idx]['h_alpha'].to_numpy())

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
        return {'label': label, 'time': time, 'h_alpha': h_window, 'path':img_path}


class TwoImagesDataset(Dataset):
    def __init__(self, annotations, img_dir, mean, std, second_img_opt: str = 'RIS2', h_alpha_window = 0):
        self.img_labels = annotations #pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.option = second_img_opt
        self.h_alpha_window = h_alpha_window

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        first_img_path = os.path.join(self.img_dir, self.img_labels.loc[idx, 'filename'])
        first_image = read_image(first_img_path).float()
        normalized_first_image = (first_image/255 - self.mean[:, None, None])/(self.std[:, None, None])
        
        #Dealing second image
        if self.option == 'RIS2':
            second_img_path = first_img_path.replace('RIS1', 'RIS2')
        elif  idx-1 < 0: #If previous img doesn't exist, use the same image twice
            second_img_path = first_img_path
        else:
            second_img_path = os.path.join(self.img_dir, self.img_labels.loc[idx-1, 'filename'])
        second_image = read_image(second_img_path).float()
        normalized_second_image = (second_image/255 - self.mean[:, None, None])/(self.std[:, None, None])

        #dealing h_alpha
        h_window = torch.tensor([])
        if idx-self.h_alpha_window < 0: 
            h_window = torch.tensor(self.h_alpha_window*[self.img_labels.iloc[idx]['h_alpha']])
        else:
            h_window = torch.tensor(self.img_labels.iloc[idx-self.h_alpha_window:idx]['h_alpha'].to_numpy())

        label = self.img_labels.iloc[idx]['mode']
        time = self.img_labels.iloc[idx]['time']
  
        labeled_imgs_dict = {'img': torch.cat((normalized_first_image.unsqueeze(0), 
                                               normalized_second_image.unsqueeze(0)), dim=0),
                             'label':label, 'path':first_img_path, 'time':time, 'h_alpha': h_window}

        return labeled_imgs_dict

    
class TwoImagesModel(nn.Module):
    """
    Initializes the TwoImagesModel composed of two pretrained resnet.
    Removes last fc layer and connects the logits with Sequential.
    Can be used for models trained RIS1 and RIS2 respectively, or for two subsequential images from RIS1

    Parameters:
    - model (torch.nn.Module): Pre-trained base model.

    - hidden_units (int): Number of hidden units in the classifier layer.
    """
    def __init__(self, modelA, modelB, hidden_units):
        super(TwoImagesModel, self).__init__()

        num_ftrs_A = modelA.fc.in_features
        num_ftrs_B = modelB.fc.in_features

        self.modelA = copy.deepcopy(modelA)
        self.modelA.fc = nn.Linear(num_ftrs_A, 256)

        self.modelB = copy.deepcopy(modelB)
        self.modelB.fc = nn.Linear(num_ftrs_B, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, hidden_units),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_units, 3) #3 for L, H, ELM
        )

    def forward(self, imgs):
        xA = self.modelA(imgs[:, 0])
        xB = self.modelB(imgs[:, 1])

        x = torch.cat((xA, xB), dim=1)

        x = self.classifier(x)
        
        
        return x   
    

def load_and_split_dataframes(path:Path, shots:list, shots_for_testing:list, shots_for_validation:list,
                            use_ELMS: bool = True):
    '''
    Takes path and lists of shots. Shots not specified in shots_for_testing
    and shots_for_validation will be used for training. Returns test_df, val_df, train_df 
    'mode' columns is then transformed to [0,1,2] notation, where 0 stands for L-mode, 1 for H-mode and 2 for ELM
    '''

    shot_df = pd.DataFrame([])

    for shot in shots:
        df = pd.read_csv(f'{path}/data/LH_alpha/LH_alpha_shot_{shot}.csv')
        df['shot'] = shot
        df = df[:-100] #!!! TODO: I cut the dataframe bacuase RIS2 sometimes ends earlier than RIS1
        shot_df = pd.concat([shot_df, df], axis=0)


    df_mode = shot_df['mode'].copy()
    df_mode[shot_df['mode']=='L-mode'] = 0
    df_mode[shot_df['mode']=='H-mode'] = 1
    df_mode[shot_df['mode']=='ELM'] = 2 if use_ELMS else 1

    shot_df['mode'] = df_mode
    shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing


    test_df = shot_df[shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
    val_df = shot_df[shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = shot_df[(~shot_df['shot'].isin(shots_for_validation))&(~shot_df['shot'].isin(shots_for_testing))].reset_index(drop=True)

    return shot_df, test_df, val_df, train_df


def get_dloader(df: pd.DataFrame(), path: Path(), batch_size: int = 32, 
                balance_data: bool = True, shuffle: bool = True,
                only_halpha: bool = False, second_img_opt: str = None, 
                h_alpha_window: int = 0, ris_option: str = 'RIS1',
                num_workers: int = 0):
    """
    Gets dataframe, path and batch size, returns "equiprobable" dataloader

    Args:
        df: should contain columns with time, confinement mode in [0,1,2] notation and filename of the images
        path: path where images are located
        batch_size: batch size
        balance_data: uses sampler if True
        shuffle: shuffles data if True
        second_img_opt: Either RIS2 image taken in the same time as RIS1, or RIS1 taken in time t-dt
        only_halpha: returns dataset with time label and h_alpha
        h_alpha_window: how many datapoints of h_alpha from previous times will be used
    Returns:  
        dataloader: dloader, which returns each class with the same probability

    Example:
    >>> get_dloader(df, path, batch_size=32, balance_data=True, shuffle=True, only_halpha=False, second_img_opt=None, h_alpha_window=0)

    """
    if ris_option == 'RIS2':
        df['filename'] = df['filename'].str.replace('RIS1', 'RIS2')

    if shuffle and balance_data:
        raise Exception("Can't use data shuffling and balancing simultaneously")
    
    if only_halpha:
        dataset = HalphaDataset(annotations=df, h_alpha_window=h_alpha_window, img_dir=path)
    elif second_img_opt == None:
        dataset = ImageDataset(annotations=df, img_dir=path, mean=mean, 
                               std=std)
    elif second_img_opt == 'RIS2':
        dataset = TwoImagesDataset(annotations=df, img_dir=path, mean=mean, std=std, 
                                    second_img_opt='RIS2', h_alpha_window=h_alpha_window)
    elif second_img_opt == 'RIS1':
        dataset = TwoImagesDataset(annotations=df, img_dir=path, mean=mean, std=std, 
                                    second_img_opt='RIS1', h_alpha_window=h_alpha_window)
    else:
        raise Exception("Can't initiate a dataset with given kwargs")

    if balance_data:
        mode_weight = (1/df['mode'].value_counts()).values
        sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
        sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                sampler=sampler, 
                                num_workers = num_workers)
    else: 
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers = num_workers)

    return dataloader


#Then calculate weights

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(inp)
    ax.grid(False)
    if title is not None:
        plt.title(title)



################### Helping functions to track the model training #############

def images_to_probs(net, batch_of_imgs):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    ''' 
    with torch.no_grad():
        output = net(batch_of_imgs)
    # convert output probabilities to predicted class
    max_logit, class_prediction = torch.max(output, 1) 
    preds = np.squeeze(class_prediction.cpu().numpy())
    return output, preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, img_paths, labels, identificator):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along   
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    modes = ['L-mode', 'H-mode', 'ELM']
    _, preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(16,9))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        image = read_image(img_paths[idx]).numpy()
        plt.grid(False)
        plt.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("Phase {3}, Prediction: {0}, {1:.1f}%\n(Label: {2})".format(
            modes[preds[idx]],
            probs[idx] * 100.0,
            modes[labels[idx]],
            identificator),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    #plt.savefig(f'{path}/preds_vs_actuals/preds_vs_actuals_{timestamp}_{identificator}.jpg')
    return fig


##################### Define model training function ##########################

def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict,
                 writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25,
                 chkpt_path=os.getcwd(), device = torch.device("cuda:0")):
    '''
    Trains the model

    Args:
        model: The neural network model to be trained.
        criterion: The loss function used for training.
        optimizer: The optimizer used for updating the model's parameters.
        scheduler: The learning rate scheduler.
        dataloaders: A dictionary containing train and validation dataloaders.
        writer: The tensorboard writer for logging.
        dataset_sizes: A dictionary containing the lengths of train and validation datasets.
        num_epochs: The number of epochs to train the model.
        chkpt_path: The path to save the best model checkpoint.
        device: The device to run the training on.

    Returns:
        The trained model.
    '''
    since = time.time()
    best_acc = 0.0

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
            num_of_samples = 0
            running_batch = 0

            # Save parameters before changing them. We will use this to calculate the average parameter change
            prev_params = {name: param.clone().detach() for name, param in model.named_parameters()}

            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                
                inputs = batch['img'].to(device).float() 
                labels = batch['label'].to(device)
                
                running_batch += 1
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) #2D tensor with shape Batchsize*len(modes)
                    #TODO: inputs.type. 
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()   
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                num_of_samples += inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) #How many correct answers
                
                
                #tensorboard part
                if running_batch % int(len(dataloaders[phase])/10)==int(len(dataloaders[phase])/10)-1: 
                    # ...log the running loss
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing loss', loss,
                                    epoch * len(dataloaders[phase]) + running_batch)
                    
                    #F1 metric
                    writer.add_scalar(f'{phase}ing F1 metric',
                                    F1Score(task="multiclass", num_classes=2).to(device)(preds, labels),
                                    epoch * len(dataloaders[phase]) + running_batch)
                    
                    #Precision recall
                    writer.add_scalar(f'{phase}ing macro Precision', 
                                        MulticlassPrecision(num_classes=2).to(device)(preds, labels),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                    writer.add_scalar(f'{phase}ing macro Recall', 
                                        MulticlassRecall(num_classes=2).to(device)(preds, labels),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                    #Average parameter change
                    # After optimizer.step(), calculate the change
                    param_change_sum = 0
                    param_count = 0
                    for name, param in model.named_parameters():
                        param_change = (param - prev_params[name]).abs().mean()
                        param_change_sum += param_change.item()
                        param_count += 1
                    avg_param_change = param_change_sum / param_count
                    
                    # Log the average parameter change
                    writer.add_scalar('Average Parameter Change', avg_param_change, 
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
        model.load_state_dict(torch.load(chkpt_path))
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    return model


####################### Test model #############

def test_model(run_path, model: torchvision.models.resnet.ResNet, 
               test_dataloader: DataLoader, max_batch: int = 0, 
               return_metrics: bool = True, comment: str ='', 
               device = torch.device("cuda:0"), writer: SummaryWriter = None):
    """
    Takes model and dataloader and returns figure with confusion matrix, 
    dataframe with predictions, F1 metric value, precision, recall and accuracy

    Args:
        model: ResNet model
        test_dataloader: DataLoader used for testing
        max_batch: maximum number of bathces to use for testing. Set = 0 to use all batches in DataLoader
        return_metrics: if True returns confusion matrix, F1, precision, recall and accuracy 
    
    Returns: 
        preds: pd.DataFrame() pd.DataFrame with columns of predicted class, true class, frame time and confidence of the prediction
        precision: MulticlassPrecision(num_classes=3)
        recall: MulticlassRecall(num_classes=3)
        accuracy: (TP+TN)/(TP+TN+FN+FP)
        fig_confusion_matrix: MulticlassConfusionMatrix(num_classes=3)
    """

    y_df = torch.tensor([])
    y_hat_df = torch.tensor([])
    preds = pd.DataFrame(columns=['shot', 'prediction', 'label', 'time', 'confidence', 'L_logit', 'H_logit', 'ELM_logit'])
    pattern = re.compile(r'RIS[12]_(\d+)_t=')
    batch_index = 0 #iterator
    for batch in tqdm(test_dataloader, desc='Processing batches'):
        batch_index +=1
        outputs, y_hat, confidence = images_to_probs(model, batch['img'].to(device).float())
        y_hat = torch.tensor(y_hat)
        y_df = torch.cat((y_df.int(), batch['label']), dim=0)
        y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
        shot_numbers = [int(pattern.search(path).group(1)) for path in batch['path']]

        pred = pd.DataFrame({'shot': shot_numbers, 'prediction': y_hat.data, 
                            'label': batch['label'].data, 'time':batch['time'], 
                            'confidence': confidence,'L_logit': outputs[:,0].cpu(), 
                            'H_logit': outputs[:,1].cpu()})

        preds = pd.concat([preds, pred],axis=0, ignore_index=True)

        if max_batch!=0 and batch_index>max_batch:
            break
    if return_metrics:
        softmax_out = torch.nn.functional.softmax(torch.tensor(preds[['L_logit', 'H_logit']].values), dim=1)
        
        # Confusion matrix
        confusion_matrix_metric = ConfusionMatrix(task="binary", num_classes=2)
        confusion_matrix_metric.update(y_hat_df, y_df)
        conf_matrix_fig, conf_matrix_ax = confusion_matrix_metric.plot()

        # F1
        f1 = F1Score(task = 'binary', num_classes=2)(y_hat_df, y_df)

        # Precision and Recall
        precision = BinaryPrecision()(y_hat_df, y_df)
        recall = BinaryRecall()(y_hat_df, y_df)

        # Precision-Recall curve
        pr_roc_auc = pr_roc_auc(y_df, softmax_out[:,1], task='binary')

        # Accuracy
        accuracy = len(preds[preds['prediction'] == preds['label']]) / len(preds)

        
        textstr = '\n'.join((
            r'for th == 0.5:',
            r'f1=%.2f' % (f1.item(), ),
            r'precision=%.2f' % (precision.item(), ),
            r'recall=%.2f' % (recall.item(), ),
            r'accuracy=%.2f' % (accuracy, )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        conf_matrix_ax.set_title(f'confusion matrix for whole test dset')

        pr_roc_auc['roc_curve'][1].text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)

        # Open the saved images using Pillow
        roc_img = matplotlib_figure_to_pil_image(pr_roc_auc['roc_curve'][0])
        conf_matrix_img = matplotlib_figure_to_pil_image(conf_matrix_fig)
        pr_curve_img = matplotlib_figure_to_pil_image(pr_roc_auc['pr_curve'][0])
        combined_image = Image.new('RGB', (conf_matrix_img.width + pr_curve_img.width + roc_img.width,\
                                            conf_matrix_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(conf_matrix_img, (0, 0))
        combined_image.paste(roc_img, (conf_matrix_img.width, 0))
        combined_image.paste(pr_curve_img, (roc_img.width+conf_matrix_img.width, 0))
        
        # Save the combined image
        combined_image.save(f'{run_path}/metrics_for_whole_test_dset_{comment}.png')

        # Save the images to tensorboard
        if writer:
            combined_image_tensor = torchvision.transforms.ToTensor()(combined_image)
            writer.add_image('metrics_for_whole_test_dset', combined_image_tensor)
                
        return {'prediction_df': preds, 'confusion_matrix': (conf_matrix_fig, conf_matrix_ax), 'f1': f1,
                'precision': precision, 'recall': recall, 'accuracy': accuracy,
                'PR_ROC_AUC': pr_roc_auc}
    else:
        return {'prediction_df': preds}
    


def per_shot_test(path, shots: list, results_df: pd.DataFrame, writer: SummaryWriter=None):
    '''
    Takes model's results dataframe from confinement_mode_classifier.test_model() and shot numbers.
    Returns metrics of model for each shot separately

    Args: 
        shots: list with numbers of shot to be tested on.
        model: ResNet model
        results_df: pd.DataFrame from confinement_mode_classifier.test_model().
        time_confidence_img: Image with model confidence on separate shot
        roc_img: Image with ROC 
        conf_matrix_img: Image with confusion matrix
        combined_image: Combined image with three previous returns
    Returns:
        path: Path where images are saved
    '''

    for shot in tqdm(shots):
        pred_for_shot = results_df[results_df['shot']==shot]
        softmax_out = torch.nn.functional.softmax(torch.tensor(pred_for_shot[['L_logit','H_logit']].values), dim=1)

        y_hat_df = torch.tensor(pred_for_shot['prediction'].values.astype(float))
        y_df = torch.tensor(pred_for_shot['label'].values.astype(int))
        
        #Confusion matrix
        confusion_matrix_metric = ConfusionMatrix(task="binary", num_classes=2)
        confusion_matrix_metric.update(y_hat_df, y_df)
        conf_matrix_fig, conf_matrix_ax = confusion_matrix_metric.plot()

        # F1
        f1 = F1Score(task = 'binary', num_classes=2)(y_hat_df, y_df)

        # Precision and Recall
        precision = BinaryPrecision()(y_hat_df, y_df)
        recall = BinaryRecall()(y_hat_df, y_df)

        #accuracy
        accuracy = len(pred_for_shot[pred_for_shot['prediction']==pred_for_shot['label']])/len(pred_for_shot)

        textstr = '\n'.join((
            f'shot {shot}',
            r'threshhold = 0.5:',
            r'f1=%.2f' % (f1.item(), ),
            r'precision=%.2f' % (precision.item(), ),
            r'recall=%.2f' % (recall.item(), ),
            r'accuracy=%.2f' % (accuracy, )))


        conf_time_fig, conf_time_ax = plt.subplots(figsize=(10,6))
        conf_time_ax.plot(pred_for_shot['time'],softmax_out[:,1], label='H-mode Confidence')

        conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==1]['time'], 
                          len(pred_for_shot[pred_for_shot['label']==1])*[1], 
                          s=2, alpha=1, label='H-mode Truth', color='maroon')



        conf_time_ax.set_xlabel('t [ms]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        conf_time_ax.text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)
        conf_time_ax.set_ylabel('Confidence')

        # Add grid to the plot
        conf_time_ax.grid(True)


        plt.title(f'shot {shot}')
        conf_time_ax.legend()
        

        conf_matrix_ax.set_title(f'confusion matrix for shot {shot}')
        conf_matrix_fig.set_figheight(conf_time_fig.get_size_inches()[1])


        # Open the saved images using Pillow
        time_confidence_img = matplotlib_figure_to_pil_image(conf_time_fig)
        conf_matrix_img = matplotlib_figure_to_pil_image(conf_matrix_fig)

        combined_image = Image.new('RGB', (time_confidence_img.width + conf_matrix_img.width,
                                            time_confidence_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(time_confidence_img, (0, 0))
        combined_image.paste(conf_matrix_img, (time_confidence_img.width, 0))
        
        # Save the combined image
        combined_image.save(f'{path}/metrics_for_shot_{shot}.png')

        # Save the images to tensorboard
        if writer:
            combined_image_tensor = torchvision.transforms.ToTensor()(combined_image)
            writer.add_image(f'metrics_for_test_shot_{shot}', combined_image_tensor)

    return f'{path}/data'

def matplotlib_figure_to_pil_image(fig):
    """
    Convert a Matplotlib figure to a PIL Image.

    Parameters:
    - fig (matplotlib.figure.Figure): The Matplotlib figure to be converted.

    Returns:
    - PIL.Image.Image: The corresponding PIL Image.

    Example:
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3, 4], [10, 5, 20, 15])
    >>> pil_image = matplotlib_figure_to_pil_image(fig)
    >>> pil_image.save("output_image.png")
    >>> pil_image.show()
    """
    # Create a FigureCanvasAgg to render the figure
    canvas = FigureCanvasAgg(fig)

    # Render the figure to a bitmap
    canvas.draw()

    # Get the RGB buffer from the bitmap
    buf = canvas.buffer_rgba()

    # Convert the buffer to a PIL Image
    image = Image.frombuffer("RGBA", canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)

    return image


def pr_roc_auc(y_true, y_pred, cmap='viridis', task = 'binary'):
    """
    Calculate the PR (Precision-Recall) and ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve)
    for a binary and ternary classification problem.

    Args:
        y_true (torch.Tensor): True labels of the binary classification problem.
        y_pred (torch.Tensor): Predicted probabilities of the positive class.

    Returns:
        dict: A dictionary containing the PR curve, ROC curve, PR AUC, and ROC AUC.

    """
    def binary_pr_roc_auc(y_true, y_pred):
        # Sort predictions and corresponding labels
        sorted_indices = torch.argsort(y_pred, descending=True)
        sorted_pred = y_pred[sorted_indices]
        sorted_true = y_true[sorted_indices]

        # Calculate TP, FP, FN for each threshold
        true_positives = torch.cumsum(sorted_true, dim=0)
        false_positives = torch.cumsum(1 - sorted_true, dim=0)
        total_positives = true_positives[-1]
        total_negatives = false_positives[-1]

        # Precision and Recall calculations
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_positives

        fpr = false_positives / total_negatives
        tpr = true_positives / total_positives

        # Calculate AUC
        auc_pr = calculate_auc(recall, precision)
        auc_roc = calculate_auc(fpr, tpr)

        return precision, recall, tpr, fpr, auc_roc, auc_pr, sorted_pred

    if task == 'binary':
        precision, recall, tpr, fpr, auc_roc, auc_pr, sorted_pred =  binary_pr_roc_auc(y_true, y_pred)

        # Create the PR and ROC curves
        fig_pr, ax_pr = plt.subplots()
        scatter_pr = ax_pr.scatter(recall, precision, c=sorted_pred, cmap=cmap, s=2)
        #ax_pr.set_xlim(0, 1)
        #ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR AUC: {auc_pr:.4f}')

        fig_roc, ax_roc = plt.subplots()
        scatter_roc = ax_roc.scatter(fpr, tpr, c=sorted_pred, cmap=cmap, s=2)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f'ROC AUC: {auc_roc:.4f}')

        # Add colorbar
        cbar_pr = fig_pr.colorbar(scatter_pr)
        cbar_pr.set_label('Threshold')

        cbar_roc = fig_roc.colorbar(scatter_roc)
        cbar_roc.set_label('Threshold')

        return {'pr_curve': (fig_pr, ax_pr), 'roc_curve': (fig_roc, ax_roc), 'pr_auc': auc_pr, 'roc_auc': auc_roc}

    elif task == 'ternary':
        cmaps = ['autumn', 'winter', 'cool']
        # Create the PR and ROC curves for L-mode
        L_precision, L_recall, L_tpr, L_fpr, L_auc_roc, L_auc_pr, L_sorted_pred =  binary_pr_roc_auc((y_true==0).long(), y_pred[:,0])
        # Create the PR and ROC curves for H-mode
        H_precision, H_recall, H_tpr, H_fpr, H_auc_roc, H_auc_pr, H_sorted_pred =  binary_pr_roc_auc((y_true==1).long(), y_pred[:,1])
        # Create the PR and ROC curves for ELM
        E_precision, E_recall, E_tpr, E_fpr, E_auc_roc, E_auc_pr, E_sorted_pred =  binary_pr_roc_auc((y_true==2).long(), y_pred[:,2])

        # Create the PR image
        fig_pr, ax_pr = plt.subplots(figsize=(10, 5))
        scatter_pr_L = ax_pr.scatter(L_recall, L_precision, c=L_sorted_pred, cmap=cmaps[0], s=2)
        scatter_pr_H = ax_pr.scatter(H_recall, H_precision, c=H_sorted_pred, cmap=cmaps[1], s=2)
        scatter_pr_E = ax_pr.scatter(E_recall, E_precision, c=E_sorted_pred, cmap=cmaps[2], s=2)
        #ax_pr.set_xlim(0, 1)
        #ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR AUC L-mode: {L_auc_pr:.4f}, H-mode: {H_auc_pr:.4f}, ELM: {E_auc_pr:.4f}')
        cbar_pr = [fig_pr.colorbar(scatter_pr_L), fig_pr.colorbar(scatter_pr_H), fig_pr.colorbar(scatter_pr_E)]
        cbar_pr[0].set_label('Threshold for L-mode')
        cbar_pr[1].set_label('Threshold for H-mode')
        cbar_pr[2].set_label('Threshold for ELM')
        cbar_pr[1].set_ticks([])
        cbar_pr[2].set_ticks([])
        # Adjust positions of colorbars in order to avoid uselessly large white space
        for i, cb in enumerate(cbar_pr):
            pos = cb.ax.get_position()
            cb.ax.set_position([0.68 - i*0.05, pos.y0, 0.02, pos.height])

        #Create the ROC image
        fig_roc, ax_roc = plt.subplots(figsize=(10, 5))
        scatter_roc_L = ax_roc.scatter(L_tpr, L_fpr, c=L_sorted_pred, cmap=cmaps[0], s=2)
        scatter_roc_H = ax_roc.scatter(H_tpr, H_fpr, c=H_sorted_pred, cmap=cmaps[1], s=2)
        scatter_roc_E = ax_roc.scatter(E_tpr, E_fpr, c=E_sorted_pred, cmap=cmaps[2], s=2)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f'ROC AUC L-mode: {L_auc_roc:.4f}, H-mode: {H_auc_roc:.4f}, ELM: {E_auc_roc:.4f}')
        cbar_roc = [fig_roc.colorbar(scatter_roc_L), fig_roc.colorbar(scatter_roc_H), fig_roc.colorbar(scatter_roc_E)]
        cbar_roc[0].set_label('Threshold for L-mode')
        cbar_roc[1].set_label('Threshold for H-mode') 
        cbar_roc[2].set_label('Threshold for ELM')
        cbar_roc[1].set_ticks([])
        cbar_roc[2].set_ticks([])
        # Adjust positions of colorbars in order to avoid uselessly large white space
        for i, cb in enumerate(cbar_roc):
            pos = cb.ax.get_position()
            cb.ax.set_position([0.68 - i*0.05, pos.y0, 0.02, pos.height])

        return {'pr_curve': (fig_pr, ax_pr), 'roc_curve': (fig_roc, ax_roc), 'pr_auc': [L_auc_pr, H_auc_pr, E_auc_pr], 'roc_auc': [L_auc_roc, H_auc_roc, E_auc_roc]}
    
    else:
        raise Exception("Task should be either 'binary' or 'ternary'")

        



def calculate_auc(x, y):
    # Convert to numpy arrays for integration
    x_np = x.numpy()
    y_np =y.numpy()

    # Use numpy's trapezoidal rule integration
    auc = np.trapz(y_np, x_np)
    return auc