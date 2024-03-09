import os
import time 
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import torch.nn.functional as F
import pandas as pd
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassConfusionMatrix, F1Score, MulticlassPrecision, MulticlassRecall, MulticlassPrecisionRecallCurve, MulticlassROC
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg

import confinement_mode_classifier as cmc

################################## Stolen from https://github.com/TheMrGhostman/InceptionTime-Pytorch/##################################
def correct_sizes(sizes):
	corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
	return corrected_sizes


def pass_through(X):
	return X


class Inception(nn.Module):
	def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(), return_indices=False):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		: param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d. 
		"""
		super(Inception, self).__init__()
		self.return_indices=return_indices
		if in_channels > 1:
			self.bottleneck = nn.Conv1d(
								in_channels=in_channels, 
								out_channels=bottleneck_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		else:
			self.bottleneck = pass_through
			bottleneck_channels = 1

		self.conv_from_bottleneck_1 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_from_bottleneck_2 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_from_bottleneck_3 = nn.Conv1d(
										in_channels=bottleneck_channels, 
										out_channels=n_filters, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
		self.conv_from_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=n_filters, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.batch_norm = nn.BatchNorm1d(num_features=4*n_filters)
		self.activation = activation

	def forward(self, X):
		# step 1
		Z_bottleneck = self.bottleneck(X)
		if self.return_indices:
			Z_maxpool, indices = self.max_pool(X)
		else:
			Z_maxpool = self.max_pool(X)
		# step 2
		Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
		Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
		Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
		Z4 = self.conv_from_maxpool(Z_maxpool)
		# step 3 
		Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
		Z = self.activation(self.batch_norm(Z))
		if self.return_indices:
			return Z, indices
		else:
			return Z


class InceptionBlock(nn.Module):
	def __init__(self, in_channels, n_filters=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU(), return_indices=False):
		super(InceptionBlock, self).__init__()
		self.use_residual = use_residual
		self.return_indices = return_indices
		self.activation = activation
		self.inception_1 = Inception(
							in_channels=in_channels,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_2 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)
		self.inception_3 = Inception(
							in_channels=4*n_filters,
							n_filters=n_filters,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation,
							return_indices=return_indices
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.Conv1d(
									in_channels=in_channels, 
									out_channels=4*n_filters, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=4*n_filters
									)
								)

	def forward(self, X):
		if self.return_indices:
			Z, i1 = self.inception_1(X)
			Z, i2 = self.inception_2(Z)
			Z, i3 = self.inception_3(Z)
		else:
			Z = self.inception_1(X)
			Z = self.inception_2(Z)
			Z = self.inception_3(Z)
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		if self.return_indices:
			return Z,[i1, i2, i3]
		else:
			return Z



class InceptionTranspose(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
		"""
		: param in_channels				Number of input channels (input features)
		: param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
		: param kernel_sizes			List of kernel sizes for each convolution.
										Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
										This is nessesery because of padding size.
										For correction of kernel_sizes use function "correct_sizes". 
		: param bottleneck_channels		Number of output channels in bottleneck. 
										Bottleneck wont be used if nuber of in_channels is equal to 1.
		: param activation				Activation function for output tensor (nn.ReLU()). 
		"""
		super(InceptionTranspose, self).__init__()
		self.activation = activation
		self.conv_to_bottleneck_1 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[0], 
										stride=1, 
										padding=kernel_sizes[0]//2, 
										bias=False
										)
		self.conv_to_bottleneck_2 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[1], 
										stride=1, 
										padding=kernel_sizes[1]//2, 
										bias=False
										)
		self.conv_to_bottleneck_3 = nn.ConvTranspose1d(
										in_channels=in_channels, 
										out_channels=bottleneck_channels, 
										kernel_size=kernel_sizes[2], 
										stride=1, 
										padding=kernel_sizes[2]//2, 
										bias=False
										)
		self.conv_to_maxpool = nn.Conv1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1, 
									stride=1,
									padding=0, 
									bias=False
									)
		self.max_unpool = nn.MaxUnpool1d(kernel_size=3, stride=1, padding=1)
		self.bottleneck = nn.Conv1d(
								in_channels=3*bottleneck_channels, 
								out_channels=out_channels, 
								kernel_size=1, 
								stride=1, 
								bias=False
								)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

		def forward(self, X, indices):
			Z1 = self.conv_to_bottleneck_1(X)
			Z2 = self.conv_to_bottleneck_2(X)
			Z3 = self.conv_to_bottleneck_3(X)
			Z4 = self.conv_to_maxpool(X)

			Z = torch.cat([Z1, Z2, Z3], axis=1)
			MUP = self.max_unpool(Z4, indices)
			BN = self.bottleneck(Z)
			# another possibility insted of sum BN and MUP is adding 2nd bottleneck transposed convolution
			
			return self.activation(self.batch_norm(BN + MUP))


class InceptionTransposeBlock(nn.Module):
	def __init__(self, in_channels, out_channels=32, kernel_sizes=[9,19,39], bottleneck_channels=32, use_residual=True, activation=nn.ReLU()):
		super(InceptionTransposeBlock, self).__init__()
		self.use_residual = use_residual
		self.activation = activation
		self.inception_1 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_2 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=in_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)
		self.inception_3 = InceptionTranspose(
							in_channels=in_channels,
							out_channels=out_channels,
							kernel_sizes=kernel_sizes,
							bottleneck_channels=bottleneck_channels,
							activation=activation
							)	
		if self.use_residual:
			self.residual = nn.Sequential(
								nn.ConvTranspose1d(
									in_channels=in_channels, 
									out_channels=out_channels, 
									kernel_size=1,
									stride=1,
									padding=0
									),
								nn.BatchNorm1d(
									num_features=out_channels
									)
								)

	def forward(self, X, indices):
		assert len(indices)==3
		Z = self.inception_1(X, indices[2])
		Z = self.inception_2(Z, indices[1])
		Z = self.inception_3(Z, indices[0])
		if self.use_residual:
			Z = Z + self.residual(X)
			Z = self.activation(Z)
		return Z
      
class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)
    
class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)
##############################################################################	

class RobustScalerNumpy:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        """
        Compute the median and IQR of X to later use for scaling.
        X should be a NumPy array.
        """
        self.median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr = q3 - q1

    def transform(self, X):
        """
        Scale features of X according to the median and IQR.
        """
        if self.median is None or self.iqr is None:
            raise RuntimeError("Must fit the scaler before transforming data.")

        # Avoid division by zero
        iqr_nonzero = np.where(self.iqr == 0, 1, self.iqr)
        return (X - self.median) / iqr_nonzero

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        """
        self.fit(X)
        return self.transform(X)


def split_df(df, shots, shots_for_testing, shots_for_validation, signal_name, use_ELMS=True, path=os.getcwd()):

    """
    Splits the dataframe into train, test and validation sets. 
    ALSO SCALES THE DATA
    """
    signal_paths_dict = {'h_alpha': f'{path}/data/h_alpha_signal', 
                         'mc': f'{path}/data/mirnov_coil_signal', 
                         'divlp': f'{path}/data/langmuir_probe_signal'}
    
    if signal_name not in ['divlp', 'mc', 'h_alpha']:
            raise ValueError(f'{signal_name} is not a valid signal name. Please use one of the following: divlp, mc, h_alpha')
        
    shot_df = pd.DataFrame([])

    for shot in shots:
        df = pd.read_csv(f'{signal_paths_dict[signal_name]}/shot_{shot}.csv')
        df['shot'] = shot
        shot_df = pd.concat([shot_df, df], axis=0)

    #Scale the data
    scaler = RobustScalerNumpy().fit_transform
    scaled_df = pd.DataFrame(scaler(shot_df[signal_name]), 
                             columns=shot_df.columns)
    scaled_df['shot'] = shot_df['shot']
    scaled_df['mode'] = shot_df['mode']
    scaled_df['time'] = shot_df['time']
    
    #Replace the mode with a number
    df_mode = scaled_df['mode'].copy()
    df_mode[scaled_df['mode']=='L-mode'] = 0
    df_mode[scaled_df['mode']=='H-mode'] = 1
    df_mode[scaled_df['mode']=='ELM'] = 2 if use_ELMS else 1
    scaled_df['mode'] = df_mode
    scaled_df = scaled_df.reset_index(drop=True) #each shot has its own indexing

    if signal_name == 'h_alpha':
         scaled_df['h_alpha']*=-1

    test_df = scaled_df[scaled_df['shot'].isin(shots_for_testing)].reset_index(drop=True)
    val_df = scaled_df[scaled_df['shot'].isin(shots_for_validation)].reset_index(drop=True)
    train_df = scaled_df[(~scaled_df['shot'].isin(shots_for_validation))&(~scaled_df['shot'].isin(shots_for_testing))].reset_index(drop=True)

    return scaled_df, test_df, val_df, train_df

class SignalDataset(Dataset):
    '''
        Parameters:
            df (DataFrame): The DataFrame containing annotations for the dataset.
            window (int): The size of the time window for fetching sequential data.

    '''

    def __init__(self, df, window, signal_name='divlp'):
        self.df = df
        self.window = window
        self.signal_name = signal_name

        if self.signal_name not in ['divlp', 'mc', 'h_alpha']:
            raise ValueError(f'{self.signal_name} is not a valid signal name. Please use one of the following: divlp, mc, h_alpha')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        signal_window = torch.tensor([])

        if idx < self.window: 
            if idx == 0: #This "if" is needed because  self.df.iloc[0:0][f'{self.signal_name}'] is a scalar => signal_window[0] will fail
                signal_window = torch.full((self.window,), self.df.iloc[0][f'{self.signal_name}'])
            else:
                signal_window = torch.tensor(self.df.iloc[0:idx][f'{self.signal_name}'].to_numpy())
                signal_window = torch.cat([torch.full((self.window-idx,), signal_window[0]), signal_window])
        else:
            signal_window = torch.tensor(self.df.iloc[idx-self.window:idx][f'{self.signal_name}'].to_numpy())

        label = self.df.iloc[idx]['mode']
        time = self.df.iloc[idx]['time']
        shot_num = self.df.iloc[idx]['shot']
        return {'label': label, 'time': time, f'{self.signal_name}': signal_window, 'shot': shot_num}
	
def get_dloader(df: pd.DataFrame(), batch_size: int = 32, 
                balance_data: bool = True, shuffle: bool = False,
                signal_window: int = 160, signal_name: str = 'divlp',
                num_workers: int = 8, persistent_workers: bool = True):
    """
    Gets dataframe, path and batch size, returns "equiprobable" dataloader

    Args:
        df: should contain columns with time, confinement mode in [0,1,2] notation and filename of the images
        batch_size: batch size
        balance_data: uses sampler if True
        shuffle: shuffles data if True
        signal_window: how many datapoints of signal from previous times will be used
    Returns:  
        dataloader: dloader, which returns each class with the same probability
    """

    if shuffle and balance_data:
        raise Exception("Can't use data shuffling and balancing simultaneously")
    
    dataset = SignalDataset(df, window=signal_window, signal_name=signal_name)

    if balance_data:
        mode_weight = (1/df['mode'].value_counts()).values
        sampler_weights = df['mode'].map(lambda x: mode_weight[x]).values
        sampler = WeightedRandomSampler(sampler_weights, len(df), replacement=True)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                sampler=sampler, 
                                num_workers=num_workers, 
                                persistent_workers=persistent_workers)
    else: 
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=num_workers, 
                                persistent_workers=persistent_workers)

    return dataloader

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=3, window=80):
        super(Simple1DCNN, self).__init__()
        # Define the 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        # Define a fully connected layer for classification
        ### in_features = floor(((input_length + 2*padding - dilation*(kernel_size - 1) - 1) // stride) + 1)
        self.fc = nn.Linear(in_features=32 * (window - 2), out_features=num_classes)

    def forward(self, x):
        # Apply 1D convolutions
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)  #!!! should I use some activation function here?

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer and return the output
        x = self.fc(x)
        return x
	

def train_model(model, criterion, optimizer, scheduler:lr_scheduler, dataloaders: dict,
                 writer: SummaryWriter, dataset_sizes={'train':1, 'val':1}, num_epochs=25,
                 chkpt_path=os.getcwd(), signal_name='divlp', device = torch.device("cuda:0")):
    since = time.time()


    torch.save(model.state_dict(), chkpt_path)
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
            #TODO: eliminate the need in that dummy iterative for tensorboard part
            for batch in tqdm(dataloaders[phase]):
                
                inputs = batch[f'{signal_name}'].to(device).float() # #TODO: is it smart to convert double to float here? 
                labels = batch['label'].to(device)
                
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

                        total_batch['train'] += 1
                        total_loss['train'] += loss.item()
                    else:
                        total_batch['val'] += 1
                        total_loss['val'] += loss.item()                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) #How many correct answers
                
                
                #tensorboard part
                
                if running_batch % int(len(dataloaders[phase])/10)==int(len(dataloaders[phase])/10)-1: 
                    # ...log the running loss
                    
                    #Training/validation loss
                    writer.add_scalar(f'{phase}ing loss', total_loss[phase] / total_batch[phase],
                                    total_batch[phase])
                    
                    #F1 metric
                    writer.add_scalar(f'{phase}ing F1 metric',
                                    F1Score(task="multiclass", num_classes=3).to(device)(preds, labels),
                                    epoch * len(dataloaders[phase]) + running_batch)
                    
                    #Precision recall
                    writer.add_scalar(f'{phase}ing macro Precision', 
                                        MulticlassPrecision(num_classes=3).to(device)(preds, labels),
                                        epoch * len(dataloaders[phase]) + running_batch)
                    
                    writer.add_scalar(f'{phase}ing macro Recall', 
                                        MulticlassRecall(num_classes=3).to(device)(preds, labels),
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

def test_model(run_path, 
               model, test_dataloader: DataLoader,
               max_batch: int = 0, return_metrics: bool = True, 
               comment: str ='', signal_name: str = 'divlp', writer: SummaryWriter = None,
               device = torch.device("cuda:0")):
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
    batch_index = 0 #iterator
    for batch in tqdm(test_dataloader, desc='Processing batches'):
        batch_index +=1
        outputs, y_hat, confidence = cmc.images_to_probs(model, batch[f'{signal_name}'].to(device).float())
        y_hat = torch.tensor(y_hat)
        y_df = torch.cat((y_df.int(), batch['label']), dim=0)
        y_hat_df = torch.cat((y_hat_df, y_hat), dim=0)
        shot_numbers = batch['shot']

        pred = pd.DataFrame({'shot': shot_numbers, 'prediction': y_hat.data, 
                            'label': batch['label'].data, 'time':batch['time'], 
                            'confidence': confidence,'L_logit': outputs[:,0].cpu(), 
                            'H_logit': outputs[:,1].cpu(), 'ELM_logit': outputs[:,2].cpu()})

        preds = pd.concat([preds, pred],axis=0, ignore_index=True)

        if max_batch!=0 and batch_index>max_batch:
            break

    if return_metrics:
        print('Processing metrics...')

        softmax_out = torch.nn.functional.softmax(torch.tensor(preds[['L_logit','H_logit','ELM_logit']].values), dim=1)

        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=3)
        confusion_matrix_metric.update(y_hat_df, y_df)
        conf_matrix_fig, conf_matrix_ax  = confusion_matrix_metric.plot()
        
        #F1
        f1 = F1Score(task="multiclass", num_classes=3)(y_hat_df, y_df)

        #Precision and recall
        precision = MulticlassPrecision(num_classes=3)(y_hat_df, y_df)
        recall = MulticlassRecall(num_classes=3)(y_hat_df, y_df)

        #Precision_recall and ROC curves are generated using the pr_roc_auc()
        pr_roc = cmc.pr_roc_auc(y_df, softmax_out, task='ternary')
        pr_fig = pr_roc['pr_curve'][0]
        roc_fig = pr_roc['roc_curve'][0]
        roc_ax = pr_roc['roc_curve'][1]

        #Accuracy
        accuracy = len(preds[preds['prediction']==preds['label']])/len(preds)

        textstr = '\n'.join((
            f'Whole test dset',
            r'threshhold = 0.5:',
            r'f1=%.2f' % (f1.item(), ),
            r'precision=%.2f' % (precision.item(), ),
            r'recall=%.2f' % (recall.item(), ),
            r'accuracy=%.2f' % (accuracy, )))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        conf_matrix_ax.set_title(f'confusion matrix for whole test dset')
        roc_ax.text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)

        # Open the saved images using Pillow
        roc_img = cmc.matplotlib_figure_to_pil_image(roc_fig)
        roc_img = roc_img.crop([int(0.06*roc_img.width), 0, int(0.8*roc_img.width), roc_img.height])
        pr_img = cmc.matplotlib_figure_to_pil_image(pr_fig)
        pr_img = pr_img.crop([int(0.06*pr_img.width), 0, int(0.8*pr_img.width), pr_img.height])
        conf_matrix_img = cmc.matplotlib_figure_to_pil_image(conf_matrix_fig)
        
        # Resize the images to have the same height
        new_conf_width = int(conf_matrix_img.width/conf_matrix_img.height * pr_img.height)
        conf_matrix_img = conf_matrix_img.resize((new_conf_width, pr_img.height))

        # Create a new image with a white background
        combined_image = Image.new('RGB', (conf_matrix_img.width + pr_img.width + roc_img.width,\
                                            roc_img.height))

        # Paste the saved images into the combined image
        combined_image.paste(conf_matrix_img, (0, 0))
        combined_image.paste(roc_img, (conf_matrix_img.width, 0))
        combined_image.paste(pr_img, (roc_img.width+conf_matrix_img.width, 0))
        
        # Save the combined image
        combined_image.save(f'{run_path}/metrics_for_whole_test_dset_{comment}.png')

        # Save the images to tensorboard
        if writer:
            combined_image_tensor = torchvision.transforms.ToTensor()(combined_image)
            writer.add_image('metrics_for_whole_test_dset', combined_image_tensor)

        return {'prediction_df': preds, 'confusion_matrix': (conf_matrix_fig, conf_matrix_ax), 'f1': f1,
                'precision': precision, 'recall': recall, 'accuracy': accuracy, 'pr_roc_curves': pr_roc}
                
    else:
        return {'prediction_df': preds}
    

def per_shot_test(path, shots: list, results_df: pd.DataFrame, writer: SummaryWriter = None):
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
        softmax_out = torch.nn.functional.softmax(torch.tensor(pred_for_shot[['L_logit','H_logit','ELM_logit']].values), dim=1)

        preds_tensor = torch.tensor(pred_for_shot['prediction'].values.astype(float))
        labels_tensor = torch.tensor(pred_for_shot['label'].values.astype(int))
        
        #Confusion matrix
        confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=3)
        confusion_matrix_metric.update(preds_tensor, labels_tensor)
        conf_matrix_fig, conf_matrix_ax = confusion_matrix_metric.plot()
        

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
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        conf_time_fig, conf_time_ax = plt.subplots(figsize=(10,6))
        conf_time_ax.plot(pred_for_shot['time'],softmax_out[:,1], label='H-mode Confidence')
        conf_time_ax.plot(pred_for_shot['time'],-softmax_out[:,2], label='ELM Confidence')

        conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==1]['time'], 
                          len(pred_for_shot[pred_for_shot['label']==1])*[1], 
                          s=2, alpha=1, label='H-mode Truth', color='maroon')
        
        conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==2]['time'], 
                          len(pred_for_shot[pred_for_shot['label']==2])*[-1], 
                          s=2, alpha=1, label='ELM Truth', color='royalblue')
    
        conf_time_ax.text(0.05, 0.3, textstr, fontsize=14, verticalalignment='bottom', bbox=props)
        conf_time_ax.set_xlabel('t [ms]')
        conf_time_ax.set_ylabel('Confidence')

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


def select_model_architecture(architecture: str, num_classes: int, window: int):
    """
    Selects and returns a model architecture based on the specified architecture name.

    Args:
        architecture (str): The name of the architecture to select. Valid options are 'InceptionTime' and 'Simple1DCNN'.
        num_classes (int): The number of output classes for the model.
        window (int): The size of the input window of given signal.

    Returns:
        torch.nn.Module: The selected model architecture.

    Raises:
        ValueError: If the specified architecture name is not valid.
    """

    if architecture == 'InceptionTime':
        model = nn.Sequential(
                    Reshape(out_shape=(1, window)),
                    InceptionBlock(
                        in_channels=1, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    InceptionBlock(
                        in_channels=32*4, 
                        n_filters=32, 
                        kernel_sizes=[5, 11, 23],
                        bottleneck_channels=32,
                        use_residual=True,
                        activation=nn.ReLU()
                    ),
                    nn.AdaptiveAvgPool1d(output_size=1),
                    Flatten(out_features=32*4*1),
                    nn.Linear(in_features=4*32*1, out_features=num_classes)
        )


    elif architecture == 'Simple1DCNN':
        model = Simple1DCNN(num_classes=num_classes, window=window)

    else:
        raise ValueError(f'{architecture} is not a valid architecture. Please use one of the following: InceptionTime, Simple1DCNN')
    return model