import os
os.chdir("/compass/Shared/Users/bogdanov/vyzkumny_ukol")
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import numpy as np         # práce s numerickými poli
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

#import module with functions from the tutorial
import cdb_img_processing as img
cdb = client.CDBClient()

def process_shots(shots):
    for shot in shots:
        print('Working on shot ', shot)
        out_path = Path('./imgs')
        ris1_data = img.load_RIS_data(shot, 1)
        ris2_data = img.load_RIS_data(shot, 2)
        ris1_names = img.save_ris_images_to_folder(ris1_data, path=out_path, ris=1, shot=shot, use_discharge_duration=True, just_names=False)
        ris2_names = img.save_ris_images_to_folder(ris2_data, path=out_path, ris=2, shot=shot, use_discharge_duration=True, just_names=False)
        LorH = pd.DataFrame(data={'mode':np.full(len(ris1_data), 'L-mode')}, index=pd.Index(ris1_data.time, name='time'))
        #contains time and the state of the plasma (L-mode, H-mode, ELM)

        t_ELM_start = cdb.get_signal(f"t_ELM_start/SYNTHETIC_DIAGNOSTICS:{shot}:bogdanov_2023")
        t_ELM_end = cdb.get_signal(f"t_ELM_end/SYNTHETIC_DIAGNOSTICS:{shot}:bogdanov_2023")
        t_H_mode_start = cdb.get_signal(f"t_H_mode_start/SYNTHETIC_DIAGNOSTICS:{shot}:bogdanov_2023")
        t_H_mode_end = cdb.get_signal(f"t_H_mode_end/SYNTHETIC_DIAGNOSTICS:{shot}:bogdanov_2023")

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


        for H_mode in t_H_mode.values:
            LorH.loc[H_mode[0]:H_mode[1]] = 'H-mode'

        for elm in t_ELM.values:
            LorH.loc[elm[0]:elm[1]] = 'ELM'

        #Discarding pictures without plasma
        discharge_start, discharge_end = img.discharge_duration(shot)
        LorH = LorH[discharge_start : discharge_end]

        #Appending columns with paths of the RIS imgs
        LorH['filename'] = np.array(ris1_names)

        LorH.to_csv(f'./LHmode-detection-shot{shot}.csv')
        print(f'csv saved to ./LHmode-detection-shot{shot}.csv')
    
if __name__ == "__main__":
    shots_str = input('input shots numbers separated with commas')
    shots = np.array([int(x) for x in shots_str.split(',')])
    process_shots(shots)
    