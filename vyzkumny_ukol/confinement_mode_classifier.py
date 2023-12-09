#import all packages and set seed
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import torchvision
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision.transforms import Normalize
import torchsummary
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from tempfile import TemporaryDirectory
from datetime import datetime
import time 
from torchmetrics.classification import MulticlassConfusionMatrix

sns.reset_orig()
sns.set()

batch_size = 32


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
# Setting the seed
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

path = Path(os.getcwd())


####################### Create datasets and dataloaders #######################

shots = [16534, 16769, 16773, 18130, 19237, 19240, 19379, 18057, 16989]
shots_for_testing = [16769, 18130, 18057]
shots_for_validation = [19237]

shot_df = pd.DataFrame([])

for shot in shots:
    df = pd.read_csv(f'{path}/LHmode-detection-shot{shot}.csv')
    df['shot'] = shot
    shot_df = pd.concat([shot_df, df], axis=0)


df_mode = shot_df['mode'].copy()
df_mode[shot_df['mode']=='L-mode'] = 0
df_mode[shot_df['mode']=='H-mode'] = 1
df_mode[shot_df['mode']=='ELM'] = 0 
shot_df['mode'] = df_mode
shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

# Images from RIS2 camera
# ris2_names = shot_df['filename'].str.replace('RIS1', 'RIS2')
# shot_df_RIS2 = shot_df.copy()
# shot_df_RIS2['filename'] = ris2_names

# Combine both datasets
#shot_df = pd.concat([shot_df, shot_df_RIS2], axis=0)
#shot_df = shot_df.reset_index(drop=True) #each shot has its own indexing

# Precalculated mean and std for each color from
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
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
        return normalized_image, label, img_path
        

#First split the dataset
test_df = shot_df[shot_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
val_df = shot_df[shot_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
train_df = shot_df[(~shot_df['shot'].isin(shots_for_validation))&(~shot_df['shot'].isin(shots_for_testing))].reset_index(drop=True)

#Then calculate weights
def get_dset(df, path, batch_size):
    """

    """
    mode_weight = (1/df['mode'].value_counts()).values
    sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
    sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
    dataset = ImageDataset(annotations=df, img_dir=path, mean=mean, std=std)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader

train_dataloader = get_dset(train_df, path=path, batch_size=batch_size)
test_dataloader = get_dset(test_df, path=path, batch_size=batch_size)
val_dataloader = get_dset(val_df, path=path, batch_size=batch_size)

dataloaders = {'train':train_dataloader, 'val':val_dataloader} #TODO: add test loader in train function and consequenlty to this dict
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(inp)
    ax.grid(False)
    if title is not None:
        plt.title(title)


modes = ['L-mode', 'H-mode', 'ELM']

################# Timestamp ###################################################
timestamp = input('add comment: ') + datetime.fromtimestamp(time.time()).strftime("-%d-%m-%Y, %H-%M-%S")#9 times!

# create grid of images
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter(f'runs/{timestamp}')

#################### Import pretrained model ##################################
pretrained_model = torchvision.models.resnet18(weights='IMAGENET1K_V1', )
num_ftrs = pretrained_model.fc.in_features
# Here the size of each output sample is set to 3.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(modes))``.
pretrained_model.fc = torch.nn.Linear(num_ftrs, 3) #3 classes: L-mode, H-mode, ELM


############# Freeze All layers except the last f.c. layer ####################

for param in pretrained_model.parameters():
    param.requires_grad = False
 
# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_ftrs, 3) #3 classes: L-mode, H-mode, ELM

#################### Define criterion and optimizer ###########################

pretrained_model = pretrained_model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.001) #pouzit adam

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

################### 

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1) 
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, img_paths, labels, identificator):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along   
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    timestamp = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y, %H-%M-%S")
    preds, probs = images_to_probs(net, images)
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
    plt.savefig(f'{path}/preds_vs_actuals/preds_vs_actuals_{timestamp}_{identificator}.jpg')
    return fig

##################### Define model training function ##########################

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, comment = ''):
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
                for inputs, labels, img_paths in tqdm(dataloaders[phase]):
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
                        
                        writer.add_scalar(f'{phase}ing loss {comment}',
                                        running_loss / num_of_samples,
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


model = train_model(pretrained_model, criterion, optimizer,
                                 exp_lr_scheduler, num_epochs=2, comment='Last f.c.')

model_path = Path(f'{path}/runs/model_last f.c._{timestamp}.pt')
torch.save(model.state_dict(), model_path)

####################### Test model with trained f.c. layer ####################

def test_model(model, test_dataloader):
    y_df = torch.tensor([])
    y_hat_df = torch.tensor([])
    wrong_preds = pd.DataFrame(columns=['path', 'Prediction', 'label'])

    for batch_index, (img, y, paths) in enumerate(test_dataloader):
        y_hat, _ = images_to_probs(model,img.float().to(device))
        y_hat = torch.tensor(y_hat)
        y_df = torch.cat((y_df, y), dim=0)
        y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)

        wrong_pred = pd.DataFrame([[paths[x], y_hat[x].item(), y[x].item()] for x in torch.where(y!=y_hat)[0]],
                                  columns=wrong_preds.columns)

        wrong_preds = pd.concat([wrong_preds, wrong_pred],axis=0, ignore_index=True)

        if batch_index>50:
            break

    metric = MulticlassConfusionMatrix(num_classes=3)
    metric.update(y_hat_df, y_df)
    fig_, ax_ = metric.plot()
    return fig_

confusion_matrix = test_model(model, test_dataloader)

writer.add_figure(f'Confusion matrix for the model with trained f.c. layer', confusion_matrix)


for param in model.parameters():
    param.requires_grad = True

model = train_model(pretrained_model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=2, comment='All layers')

model_path = Path(f'{path}/runs/model_{timestamp}_with_all_weights_trained.pt')
torch.save(model.state_dict(), model_path)


###################### Test the model with all weights trained ################

y_df = torch.tensor([])
y_hat_df = torch.tensor([])
wrong_preds = pd.DataFrame(columns=['path', 'Prediction', 'label'])

for batch_index, (img, y, paths) in enumerate(test_dataloader):
    y_hat, _ = images_to_probs(model,img.float().to(device))
    y_hat = torch.tensor(y_hat)
    y_df = torch.cat((y_df, y), dim=0)
    y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
    
    wrong_pred = pd.DataFrame([[paths[x], y_hat[x].item(), y[x].item()] for x in torch.where(y!=y_hat)[0]],
                              columns=wrong_preds.columns)
    
    wrong_preds = pd.concat([wrong_preds, wrong_pred],axis=0, ignore_index=True)
    
    if batch_index>50:
        break

metric = MulticlassConfusionMatrix(num_classes=3)
metric.update(y_hat_df, y_df)
fig_, ax_ = metric.plot()


