import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import pandas as pd
import torchvision
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision.transforms import Normalize
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall, MulticlassPrecisionRecallCurve, MulticlassROC
import torchsummary
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from tempfile import TemporaryDirectory
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time 

sns.reset_orig()
sns.set()



# Setting the seed
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



####################### Create datasets and dataloaders #######################

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def load_and_split_dataframes(path:Path, shots:list, shots_for_testing:list, shots_for_validation:list, use_ris2:bool = False):
    '''
    Takes path and lists of shots. Shots not specified in shots_for_testing
    and shots_for_validation will be used for training. Returns test_df, val_df, train_df 
    'mode' columns is then transformed to [0,1,2] notation, where 0 stands for L-mode, 1 for H-mode and 2 for ELM
    '''

    shot_df = pd.DataFrame([])

    for shot in shots:
        df = pd.read_csv(f'{path}/data/LHmode-detection-shots/LHmode-detection-shot{shot}.csv')
        df['shot'] = shot
        shot_df = pd.concat([shot_df, df], axis=0)


    df_mode = shot_df['mode'].copy()
    df_mode[shot_df['mode']=='L-mode'] = 0
    df_mode[shot_df['mode']=='H-mode'] = 1
    df_mode[shot_df['mode']=='ELM'] = 0 
    shot_df['mode'] = df_mode
    shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

    if use_ris2:
        ris2_names = shot_df['filename'].str.replace('RIS1', 'RIS2')
        shot_df_RIS2 = shot_df.copy()
        shot_df_RIS2['filename'] = ris2_names

        #Combine both datasets
        shot_df = pd.concat([shot_df, shot_df_RIS2], axis=0)
        shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

    test_df = shot_df[shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
    val_df = shot_df[shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = shot_df[(~shot_df['shot'].isin(shots_for_validation))&(~shot_df['shot'].isin(shots_for_testing))].reset_index(drop=True)

    return shot_df, test_df, val_df, train_df

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
        return normalized_image, label, img_path, time


#Then calculate weights
def get_dloader(df: pd.DataFrame(), path: Path(), batch_size: int = 32, balance_data: bool = True, shuffle: bool = True):
    """
    Gets dataframe, path and batch size, returns "equiprobable" dataloader

    Args:
        df: should contain columns with time, confinement mode in [0,1,2] notation and filename of the images
        path: path where images are located
        batch_size: batch size
    Returns: 
        dataloader: dloader, which returns each class with the same probability
    """

    dataset = ImageDataset(annotations=df, img_dir=path, mean=mean, std=std)

    if balance_data:
        mode_weight = (1/df['mode'].value_counts()).values
        sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
        sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else: 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


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

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    with torch.no_grad():
        output = net(images)
    # convert output probabilities to predicted class
    max_logit, class_prediction = torch.max(output, 1) 
    preds = np.squeeze(class_prediction.cpu().numpy())
    return output, preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, img_paths, labels, identificator, path):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along   
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    modes = ['L-mode', 'H-mode', 'ELM']
    timestamp = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y, %H-%M-%S")
    _, preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(16,9))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        image = read_image(img_paths[idx]).numpy()
        plt.grid(False)
        plt.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title("Prediction: {0}, {1:.1f}%\n(Label: {2})".format(
            modes[preds[idx]],
            probs[idx] * 100.0,
            modes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    #plt.savefig(f'{path}/preds_vs_actuals/preds_vs_actuals_{timestamp}_{identificator}.jpg')
    return fig

##################### Define model training function ##########################

def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict, writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25, comment = ''):
    '''
    Trains the model

    Args:
        model: 
        criterion: 
        optimizer: 
        scheduler: 
        num_epochs: 
        comment: 
        dataloaders: dictionary containing train and validation dataloaders
        dataset_sizes: dictionary containing lengths of train and validation datasets
        writer: tensorboard writer
    '''
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
    
        torch.save(model.state_dict(), best_model_params_path)
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
                # Iterate over data.
                #TODO: eliminate the need in that dummy iterative for tensorboard part
                for inputs, labels, img_paths, times in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device).float() # #TODO: is it smart to convert double to float here? 
                    labels = labels.to(device)
                    
                    running_batch += 1
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) #2D tensor with shape Batchsize*len(modes)
                        #TODO: inputs.type. 
                        _, preds = torch.max(outputs, 1) #preds = 1D array of indicies of maximum values in row. ([2,1,2,1,2]) - third feature is largest in first sample, second in second...
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
                        writer.add_scalar(f'{phase}ing loss {comment}',
                                        running_loss / num_of_samples,
                                        epoch * len(dataloaders[phase]) + running_batch)
                        
                        #F1 metric
                        writer.add_scalar(f'{phase}ing F1 metric {comment}',
                                        F1Score(task="multiclass", num_classes=3).to(device)(preds, labels),
                                        epoch * len(dataloaders[phase]) + running_batch)
                        
                        #Precision recall
                        writer.add_scalar(f'{phase}ing macro Precision {comment}', 
                                          MulticlassPrecision(num_classes=3).to(device)(preds, labels),
                                          epoch * len(dataloaders[phase]) + running_batch)
                        
                        writer.add_scalar(f'{phase}ing macro Recall {comment}', 
                                          MulticlassRecall(num_classes=3).to(device)(preds, labels),
                                          epoch * len(dataloaders[phase]) + running_batch)
                        
                        
                    
                    if running_batch % int(len(dataloaders[phase])/3)==int(len(dataloaders[phase])/3)-1:
                        # ...log a Matplotlib Figure showing the model's predictions on a
                        # random mini-batch
                        writer.add_figure(f'predictions vs. actuals {comment}',
                                        plot_classes_preds(model, inputs, img_paths, labels, identificator=phase),
                                        global_step=epoch * len(dataloaders[phase]) + running_batch)
                        writer.close()
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    writer.add_scalar(f'best_accuracy for epoch {comment}',
                                        epoch_acc,
                                        epoch)
                    writer.close()
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


####################### Test model with trained f.c. layer ####################

def test_model(model: torchvision.models.resnet.ResNet, test_dataloader: DataLoader, max_batch: int = 0, return_metrics: bool = True):
    '''
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
    '''
    y_df = torch.tensor([])
    y_hat_df = torch.tensor([])
    preds = pd.DataFrame(columns=['shot', 'prediction', 'label', 'time', 'confidence', 'L_logit', 'H_logit', 'ELM_logit'])
    pattern = re.compile(r'RIS1_(\d+)_t=')
    
    for batch_index, (img, y, paths, times) in tqdm(enumerate(test_dataloader)):
        outputs, y_hat, confidence = images_to_probs(model,img.float().to(device))
        y_hat = torch.tensor(y_hat)
        y_df = torch.cat((y_df.int(), y), dim=0)
        y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
        shot_numbers = [int(pattern.search(path).group(1)) for path in paths]
        pred = pd.DataFrame({'shot': shot_numbers, 'prediction': y_hat.data, 'label': y.data, 'time':times, 'confidence': confidence,
                            'L_logit': outputs[:,0].cpu(), 'H_logit': outputs[:,1].cpu(), 'ELM_logit': outputs[:,2].cpu()})

        preds = pd.concat([preds, pred],axis=0, ignore_index=True)

        if max_batch!=0 and batch_index>max_batch:
            break

    if return_metrics:
        softmax_out = torch.nn.functional.softmax(torch.tensor(preds[['L_logit','H_logit','ELM_logit']].values), dim=1)
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=3)
        confusion_matrix_metric.update(y_hat_df, y_df)
        confusion_matrix = confusion_matrix_metric.plot()

        #F1
        f1 = F1Score(task="multiclass", num_classes=3)(y_hat_df, y_df)

        #Precision
        precision = MulticlassPrecision(num_classes=3)(y_hat_df, y_df)
        recall = MulticlassRecall(num_classes=3)(y_hat_df, y_df)
        #precision(logits_df, y_df.int())
         #Precision_recall curve
        pr_curve = MulticlassPrecisionRecallCurve(num_classes=3, thresholds=64)
        pr_curve.update(softmax_out, y_df)
        pr_curve_fig, pr_curve_ax = pr_curve.plot(score=True)
        #ROC metric
        mcroc = MulticlassROC(num_classes=3, thresholds=64)
        mcroc.update(torch.tensor(preds[['L_logit', 'H_logit', 'ELM_logit']].values.astype(float)), y_df)
        roc_fig, roc_ax = mcroc.plot(score=True)
        #Accuracy
        accuracy = len(preds[preds['prediction']==preds['label']])/len(preds)
    else: 
        confusion_matrix, f1, precision, recall, accuracy = None, None, None, None, None
    return preds, confusion_matrix, f1, precision, recall, accuracy, (pr_curve_fig, pr_curve_ax), (roc_fig, roc_ax)


def per_shot_test(path, shots: list, results_df: pd.DataFrame):
    '''
    Takes model, its result dataframe from confinement_mode_classifier.test_model() and shot numbers.
    Returns metrics of model for different shots separately

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
        softmax_out = torch.nn.functional.softmax(torch.tensor(pred_for_shot[['L_logit','H_logit','ELM_logit']].values), dim=1)

        preds_tensor = torch.tensor(pred_for_shot['prediction'].values.astype(float))
        labels_tensor = torch.tensor(pred_for_shot['label'].values.astype(int))
        
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=3)
        confusion_matrix_metric.update(preds_tensor, labels_tensor)
        conf_matrix_fig, conf_matrix_ax = confusion_matrix_metric.plot()
        
        #Precision_recall curve
        pr_curve = MulticlassPrecisionRecallCurve(num_classes=3, thresholds=64)
        pr_curve.update(softmax_out, labels_tensor)
        pr_curve_fig, pr_curve_ax = pr_curve.plot(score=True)

        #ROC metric
        mcroc = MulticlassROC(num_classes=3, thresholds=64)
        mcroc.update(torch.tensor(pred_for_shot[['L_logit', 'H_logit', 'ELM_logit']].values.astype(float)), labels_tensor)
        roc_fig, roc_ax = mcroc.plot(score=True)

        #f1 score
        f1 = F1Score(task="multiclass", num_classes=3)(preds_tensor, labels_tensor)

        #Precision
        precision = MulticlassPrecision(num_classes=3)(preds_tensor, labels_tensor)

        #recall
        recall = MulticlassRecall(num_classes=3)(preds_tensor, labels_tensor)

        #accuracy
        accuracy = len(pred_for_shot[pred_for_shot['prediction']==pred_for_shot['label']])/len(pred_for_shot)

        textstr = '\n'.join((
            f'shot {shot}',
            r'threshhold = 0.5:',
            r'f1=%.2f' % (f1.item(), ),
            r'precision=%.2f' % (precision.item(), ),
            r'recall=%.2f' % (recall.item(), ),
            r'accuracy=%.2f' % (accuracy, )))


        fig, ax = plt.subplots(figsize=(10,6))

        ax.plot(pred_for_shot['time'],softmax_out[:,1], label='model confidence')
        ax.plot(pred_for_shot['time'],pred_for_shot['label'], lw=3, alpha=.5, label='label')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('H-mod confidence')
        plt.title(f'shot {shot}')
        ax.legend()

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        roc_ax.text(0.05, 0.3, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', bbox=props)
        conf_matrix_ax.set_title(f'confusion matrix for shot {shot}')
        pr_curve_ax.set_title(f'pr_curve for shot {shot}')
        
        conf_matrix_fig.set_figheight(fig.get_size_inches()[1])
        # Save the figures to temporary files
        fig.savefig(f'{path}/data/time_confidence_for_shot_{shot}.png')
        roc_fig.savefig(f'{path}/data/roc_for_shot_{shot}.png')
        conf_matrix_fig.savefig(f'{path}/data/confusion_matrix_for_shot_{shot}.png')
        pr_curve_fig.savefig(f'{path}/data/pr_curve_for_shot_{shot}.png')

        # Open the saved images using Pillow
        time_confidence_img = Image.open(f'{path}/data/time_confidence_for_shot_{shot}.png')
        roc_img = Image.open(f'{path}/data/roc_for_shot_{shot}.png')
        conf_matrix_img = Image.open(f'{path}/data/confusion_matrix_for_shot_{shot}.png')
        pr_curve_img = Image.open(f'{path}/data/pr_curve_for_shot_{shot}.png')

        combined_image = Image.new('RGB', (time_confidence_img.width + conf_matrix_img.width,\
                                            time_confidence_img.height + roc_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(time_confidence_img, (0, 0))
        combined_image.paste(conf_matrix_img, (time_confidence_img.width, 0))
        combined_image.paste(roc_img, (0, time_confidence_img.height))
        combined_image.paste(pr_curve_img, (roc_img.width, time_confidence_img.height))
        
        # Save the combined image
        combined_image.save(f'{path}/data/combined_image_for_shot_{shot}.png')

    return f'{path}/data'

