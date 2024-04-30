import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, Arc
from IPython.display import display, clear_output
from matplotlib.image import imread
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image



def visualize(path_to_run, shot,  figure_vertical_size, figure_horizontal_size, zoom_signal, zoom_time, time_for_signal):
    clear_output()
    try:
        with open(f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/runs/{path_to_run}/hparams.json', 'r') as f:
            hparams = json.load(f)
    except FileNotFoundError:
        print(f"Can't find the hparams. Chosen run is either irrelevant or the model is still training.")
        fig = shot_not_present(color='red')
        plt.close(fig)
        display(fig)
        return
    
    if 'last_fc' in path_to_run:
        print(f"I don't save the prediction_df.csv for RIS models with only last fc trained, as it's always worse than the fully trained model.")
        fig = shot_not_present(color='red')
        plt.close(fig)
        display(fig)
        return

    preds_csv = pd.read_csv(f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/runs/{path_to_run}/prediction_df.csv')
    metrics_per_shot = pd.read_csv(f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/runs/{path_to_run}/metrics_per_shot.csv')


    if not shot in hparams['shots_for_testing']:
        print(f'Shot {shot} was not (yet?) processed in this run.')
        print(f'Model is either still training, or shot {shot} was not in the testing set.')
        fig = shot_not_present(shot)
        plt.close(fig)
        display(fig)
        return
    
    #Find the shot in the results_df
    pred_for_shot = preds_csv[preds_csv['shot']==shot]

    if len(pred_for_shot)==0:
        print(f'Shot {shot} not found in the results_df. Probably L-mode was removed from the dataset, and shot {shot} was L-mode only shot.')
        fig = shot_not_present(shot)
        plt.close(fig)
        display(fig)
        return

    #Zoom in the time
    pred_for_shot = pred_for_shot[(pred_for_shot['time'] >= time_for_signal - exp_decaying(zoom_time)) & (pred_for_shot['time'] <= time_for_signal + exp_decaying(zoom_time))] #Zoom in the time


    #calculate "confidences" for the model
    if 'ELM_logit' in pred_for_shot.columns:
        softmax_out = torch.nn.functional.softmax(torch.tensor(pred_for_shot[['L_logit','H_logit','ELM_logit']].values), dim=1)
    else:
        softmax_out = torch.nn.functional.softmax(torch.tensor(pred_for_shot[['L_logit','H_logit']].values), dim=1)

    #Get the metrics for the shot
    kappa, f1, precision, recall = metrics_per_shot[metrics_per_shot['shot']==shot][['kappa','f1','precision','recall']].values[0]

    #Create the main figure - confidence over time
    conf_time_fig, conf_time_ax = plt.subplots(figsize=(10*figure_horizontal_size,6*figure_vertical_size))

    #If the model takes data from the shot twice (one for each RIS), then split the data. 
    if 'ris_option' in hparams and hparams['ris_option']=='both': 
        #I assume, that the first half of the data is from RIS1 and the second half from RIS2. 
        #It should be by how the split_df() function works
        pred_for_shot_ris1, pred_for_shot_ris2 = split_into_two_monotonic_dfs(pred_for_shot)

        #Plot confidence for the first class (it may be H-mode or ELM, depending on the model)
        conf_time_ax.plot(pred_for_shot_ris1['time'],softmax_out[:len(pred_for_shot_ris1),1], 
                          label='1st class Confidence RIS1', alpha=0.5)
        conf_time_ax.plot(pred_for_shot_ris2['time'],softmax_out[len(pred_for_shot_ris1):,1], 
                          label='1st class Confidence RIS2', alpha=0.5)
        
        conf_time_ax.scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==1]['time'], 
                      len(pred_for_shot_ris1[pred_for_shot_ris1['label']==1])*[1], 
                      s=10, alpha=1, label='1st class Truth', color='maroon')
        
        conf_time_ax.vlines(time_for_signal, 0, 1, color='black', linestyle='--')


        conf_time_ax.set_title(f'Shot {shot}, RIS1/RIS2: kappa = {kappa:.2f}, F1 = {f1:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}')

        #Plot confidence for the second class (it's alway ELM)
        if 'ELM_logit' in pred_for_shot.columns:
            conf_time_ax.plot(pred_for_shot_ris1['time'],-softmax_out[:len(pred_for_shot_ris1),2], 
                              label='2d class Confidence RIS1', alpha=0.5)
            
            conf_time_ax.plot(pred_for_shot_ris2['time'],-softmax_out[len(pred_for_shot_ris1):,2], 
                              label='2d class Confidence RIS2', alpha=0.5)
            
            conf_time_ax.scatter(pred_for_shot_ris1[pred_for_shot_ris1['label']==2]['time'], 
                len(pred_for_shot_ris1[pred_for_shot_ris1['label']==2])*[-1], 
                s=10, alpha=1, label='ELM Truth', color='royalblue')

    #If the model takes data from the shot once, then plot the confidence directly
    else:
        conf_time_ax.plot(pred_for_shot['time'],softmax_out[:,1], label='1st class Confidence')

        conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==1]['time'], 
                        len(pred_for_shot[pred_for_shot['label']==1])*[1], 
                        s=10, alpha=1, label='1st class Truth', color='maroon')

        conf_time_ax.set_title(f'Cohen kappa = {kappa:.2f}, F1 = {f1:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}')
        conf_time_ax.vlines(time_for_signal, 0, 1, color='black', linestyle='--')

        if 'ELM_logit' in pred_for_shot.columns:
            conf_time_ax.plot(pred_for_shot['time'],-softmax_out[:,2], label='2d class Confidence')

            conf_time_ax.scatter(pred_for_shot[pred_for_shot['label']==2]['time'], 
                            len(pred_for_shot[pred_for_shot['label']==2])*[-1], 
                            s=10, alpha=1, label='ELM Truth', color='royalblue')

    conf_time_ax.set_xlabel('Time [ms]')
    conf_time_ax.set_ylabel('Confidence')

    

    ###Handle the second plot. It is either a signal or an image
    #If it's an image, then load both images for RIS1/2 and display it. Hope that user will know which image the given model used
    if 'ris_option' in hparams:
        closest_time_str, closest_time = closest_decimal_time(time_for_signal)
        image1 = imread(f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/imgs/{shot}/RIS1_{shot}_t={closest_time_str}.png')
        image2 = imread(f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/imgs/{shot}/RIS2_{shot}_t={closest_time_str}.png')

        signal_fig, signal_axs = plt.subplots(1, 2, figsize=(10*figure_horizontal_size, 5*figure_vertical_size))
        
        for i, [ax, img] in enumerate(zip(signal_axs, [image1, image2]), 1):
            ax.imshow(img)
            ax.set_title(f'RIS{i}')
            ax.axis('off')  # Turn off axes for all images in one go
        signal_fig.suptitle(f'Shot {shot}, time {closest_time} ms, GT class: {pred_for_shot[pred_for_shot["time"]==closest_time]["label"].values[0]}')
        
        #plt.close(img_fig)

    #If it's a signal, then load the signal and display it
    else:
        closest_time = pred_for_shot.iloc[(pred_for_shot['time'] - time_for_signal).abs().argmin()]
        signal_paths_dict = {'h_alpha': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/h_alpha_signal_{hparams["sampling_frequency"]}kHz', 
                            'mc': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz',
                            'mcDIV': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcHFS': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcLFS': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'mcTOP': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_coil_signal_{hparams["sampling_frequency"]}kHz', 
                            'divlp': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/langmuir_probe_signal_{hparams["sampling_frequency"]}kHz',
                            'mc_h_alpha': f'/compass/Shared/Users/bogdanov/vyzkumny_ukol/data/mirnov_h_alpha_signal_{hparams["sampling_frequency"]}kHz'}
        
        signal_df = pd.read_csv(f'{signal_paths_dict[hparams["signal_name"]]}/shot_{shot}.csv')
        signal_df = signal_df[(signal_df['time'] >= time_for_signal - exp_decaying(zoom_time)) & (signal_df['time'] <= time_for_signal + exp_decaying(zoom_time))]
        signal_columns = [col for col in signal_df.columns if col not in ['time', 'mode']]
        
        #If the model is trained on only one signal, then plot it directly
        if len(signal_columns) == 1:
            percentile = signal_df[signal_columns[0]].quantile(exp_decaying(zoom_signal)/1000)
            signal_fig = plt.figure(figsize=(10*figure_horizontal_size,6*figure_vertical_size))
            signal_ax = signal_fig.add_subplot(1, 1, 1)
            signal_ax.plot(signal_df['time'], signal_df[signal_columns[0]])
            signal_ax.set_title(signal_columns[0])
            signal_ax.set_ylabel(f'{signal_columns[0]}')
            signal_ax.set_xlabel('Time [ms]')
            signal_ax.set_ylim(-percentile, percentile)

            #Plot the signal window on signal figure
            signal_ax.vlines(time_for_signal-hparams['signal_window']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--', label='Signal window')
            signal_ax.vlines(time_for_signal+hparams['signal_window']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--')
            #Plot the signal window on main figure
            conf_time_ax.vlines(time_for_signal-hparams['signal_window']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--', label='Signal window')
            conf_time_ax.vlines(time_for_signal+hparams['signal_window']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--')
            
            plt.close(signal_fig)

        #Else plot all signals (There may be 4 of them)
        else:
            signal_fig, signal_axs = plt.subplots(len(signal_columns), figsize=(10*figure_horizontal_size,6*figure_vertical_size), sharex=True)
            for signal_ax, col in zip(signal_axs, signal_columns):
                percentile = signal_df[col].quantile(exp_decaying(zoom_signal)/1000)
                tretile = signal_df[col].quantile(0.33)
                # Assuming you have data to plot related to 'col'
                signal_ax.plot(signal_df['time'], signal_df[col])
                signal_ax.set_ylabel(f'{col}')
                #Plot the signal window on signal figure
                signal_ax.vlines(time_for_signal-hparams['signal_window']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--', label='Signal window')
                signal_ax.vlines(time_for_signal+hparams['signal_window']/hparams['sampling_frequency'], 0, percentile, color='green', linestyle='--')
                signal_ax.set_ylim(-percentile, percentile)
                if col=='h_alpha':
                    signal_ax.set_ylim(tretile, percentile)
                else:
                    signal_ax.set_ylim(-percentile, percentile)
            #Plot the signal window on main figure
            conf_time_ax.vlines(time_for_signal-hparams['signal_window']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--', label='Signal window')
            conf_time_ax.vlines(time_for_signal+hparams['signal_window']/hparams['sampling_frequency'], 0, 1, color='green', linestyle='--')
            signal_axs[-1].set_xlabel('Time [ms]')    
            plt.close(signal_fig)
        

        conf_time_ax.legend()
        plt.close(conf_time_fig)
        plt.close(signal_fig)

    
    

def exp_decaying(x):
    return 1000*np.exp(-0.06908*x)

def shot_not_present(shot=False, color='yellow'):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Drawing a sad face
    # Face outline
    face = Circle((0.5, 0.5), 0.4, color=color, fill=True)
    ax.add_patch(face)

    # Eyes
    left_eye = Circle((0.35, 0.65), 0.05, color='black', fill=True)
    right_eye = Circle((0.65, 0.65), 0.05, color='black', fill=True)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    # Sad mouth
    mouth = Arc((0.5, 0.45), 0.2, 0.2, angle=0, theta1=0, theta2=180, color='black', linewidth=2)
    ax.add_patch(mouth)

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Remove axes
    ax.axis('off')

    # Set title
    if not shot:
        ax.set_title('No results available for this run.')
    else:
        ax.set_title(f'Shot {shot} not present in results for this run.')

    # Show the plot
    return fig


def closest_decimal_time(img_time):
    # Allowed decimal parts
    decimal_parts = [0, 0.2, 0.4, 0.6, 0.8]

    # Extract the integer part
    integer_part = int(img_time)

    # Find the closest decimal part
    closest_decimal = min(decimal_parts, key=lambda x: abs(x - (img_time - integer_part)))

    # Combine the integer and closest decimal part
    closest_time = integer_part + closest_decimal
    return f'{closest_time:.1f}', closest_time

def split_into_two_monotonic_dfs(df):
    """
    We have a df with non monotonic time column.
    Split it into two monotonic dfs.

    Parameters:
    df (pd.DataFrame): DataFrame with a 'time' column.

    Returns:
    tuple of pd.DataFrame: Two DataFrames split at the first non-monotonic point.
    """
    # Find the first non-monotonic increase point
    for i in range(1, len(df)):
        if df['time'].iloc[i] <= df['time'].iloc[i - 1]:
            # Split the DataFrame at the found index
            df1 = df.iloc[:i]
            df2 = df.iloc[i:]
            return df1, df2

    # If no non-monotonic point found, return the entire DataFrame as one part, and an empty DataFrame as the other
    return df, pd.DataFrame(columns=df.columns)
