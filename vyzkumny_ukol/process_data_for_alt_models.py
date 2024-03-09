import os

from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import time 
import re
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import pandas as pd
import matplotlib.pyplot as plt
import imgs_processing as imgs
import numpy as np
from tqdm.auto import tqdm
cdb = client.CDBClient()

path = Path('/compass/Shared/Users/bogdanov/vyzkumny_ukol')
os.chdir(path)

def process_data_for_alt_models(shot_numbers, variant = 'seidl_2023'):
    #dirs where to save the csv files
    directories = {'h_alpha':'data/h_alpha_signal', 
                    'mc':'data/mirnov_coil_signal', 
                    'divlp':'data/langmuir_probe_signal'}

    for shot in tqdm(shot_numbers):
        print('working on shot:', shot)

        # Load signals from CDB
        h_alpha_signal = cdb.get_signal(f"H_alpha/SPECTROMETRY_RAW:{shot}")
        mc_signal = cdb.get_signal(f"Mirnov_coil_A_theta_13_RAW/MAGNETICS_RAW:{shot}")
        divlp_signal = cdb.get_signal(f"DIVLPB01/STRATUS:{shot}")

        # Load labels from CDB
        t_ELM_start = cdb.get_signal(f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
        t_ELM_end = cdb.get_signal(f"t_ELM_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
        t_H_mode_start = cdb.get_signal(f"t_H_mode_start/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")
        t_H_mode_end = cdb.get_signal(f"t_H_mode_end/SYNTHETIC_DIAGNOSTICS:{shot}:{variant}")

        #TODO:  To create a DataFrame with only one row, one needs to specify an index, 
        # so if plasma enters H-mode more than once during one shot index have to be passed. Thus crutch with try: except:
        try:
            len(t_ELM_start.data)
        except:
            t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end.data}, index=[0])
        else:
            t_ELM = pd.DataFrame({'start':t_ELM_start.data, 'end':t_ELM_end.data})

        try:
            len(t_H_mode_start.data)
        except:
            t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data}, index=[0])
        else:
            t_H_mode = pd.DataFrame({'start':t_H_mode_start.data, 'end':t_H_mode_end.data})


        for signal, signal_name in zip([h_alpha_signal, mc_signal, divlp_signal], ['h_alpha', 'mc', 'divlp']):
            signal_df = pd.DataFrame({'time':signal.time_axis.data, signal_name:signal.data})

            # Set time as index
            signal_df = signal_df.set_index('time')

            ### Downsample data
            #First define desired frequency
            desired_frequency = 300 #in kHz
            time_resolution = 1/desired_frequency
            # Find how many rows do we need to skip to get the desired frequency 
            #(raw data have different frequencies for different shots)
            skip_rows = 0
            while signal_df.index[skip_rows] - signal_df.index[0]  < time_resolution:
                skip_rows += 1

            # Downsample
            signal_df = signal_df.iloc[::skip_rows]

            # Remove data with no plasma
            discharge_start, discharge_end = imgs.discharge_duration(shot, 4e4)
            signal_df = signal_df[np.logical_and(signal_df.index>discharge_start, signal_df.index<discharge_end)]

            # Create a column with mode labels. These are all L-mode by default.
            signal_df['mode'] = 'L-mode'

            for H_mode in t_H_mode.values:
                signal_df.loc[H_mode[0]:H_mode[1], 'mode'] = 'H-mode'

            for elm in t_ELM.values:
                signal_df.loc[elm[0]:elm[1], 'mode'] = 'ELM'

            # Save data
            signal_df.to_csv(f'{directories[signal_name]}/shot_{shot}.csv')

if __name__ == "__main__":
    #dirs to find the shot numbers. This is not used in the code, but it is useful to know the shot numbers
    data_dir_path = f'{path}/data/LH_alpha'
    file_names = os.listdir(data_dir_path)
    shot_numbers = [int(re.search(r'shot_(\d+)', file_name).group(1)) for file_name in file_names]
    shot_numbers.remove(17848) # this shot has no langmuir probe data
    
    process_data_for_alt_models(shot_numbers, variant='seidl_2023')
    
    