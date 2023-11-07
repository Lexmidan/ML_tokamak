from typing import Tuple   # v moderním pythonu lze nezávazně specifikovat typ argumentů funkce
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import numpy as np         # práce s numerickými poli
from pathlib import Path   # reprezentace cest v souborovém systému
import xarray as xr        # data s osami
from PIL import Image


def load_RIS_data(shot: int, ris: int) -> xr.DataArray:
    """Load RAW camera data from database

    Args:
        shot: Shot number.
        ris: Which camrea to load? 1 or 2.

    Returns:
        Xarray DataArray of shape (time, 2*nx, 2*ny).

    Raises:
        KeyError: When the data is missing for the shot.
        CDBException: Thrown by the database client.
        ValueError: When the camera number is incorrect.
    """
    if ris not in [1, 2]:
        raise ValueError(f'RIS camera nr. {ris} does not exist')
    s = cdbxr.Shot(shot, cache=False)

    return s[f'RIS.RISEye_{ris}.RAW']

def demosaic(data: xr.DataArray) -> np.array:
    """Demosaic RAW camera data

    Transforms RAW data of shape (time, 2*nx, 2*ny) to (time, nx, ny, 3) and normalizes them into range <0, 1>.

    Args:
        data: RAW camera data

    Returns:
        Data in RGB format.
    """

    r= data[:,::2,1::2]
    b= data[:,1::2,::2]
    g1=data[:,1::2,1::2]
    g2=data[:,::2,::2]
    g=xr.DataArray(g1.data+g2.data) #don't wanna lose any information. Green will be normalized anyway

    xar=xr.DataArray([r,g,b], dims=('color', 'time', 'x', 'y'), coords={'color': ['r','g','b'], 'time': data.coords['time'],\
                                                                         'x': np.arange(r.shape[1]), 'y':np.arange(r.shape[2])})
    axr=xar.transpose('time', 'x', 'y', 'color')
    np_axr = axr.data.astype('float64') 

    #Normalizace
    channel_min = np.min(np_axr, axis=(0, 1, 2))  # minimum value for each time and channel
    channel_span = np.max(np_axr, axis=(0, 1, 2)) - channel_min  # span of values for each time and channel
    channel_norm = np.where(channel_span > 0, channel_span, 1)  # use 1 when span is 0
    axr = (axr - channel_min[None, None, :]) / channel_norm[None, None, :]
    np_axr=axr.data

    return np_axr, axr

def flip_image(data: xr.DataArray, flip_horizontal: bool = True, flip_vertical: bool = True) -> np.array:
    """Flip image data vertically and horizontally.

    Args:
        data: Image data with shape (time, nx, ny, 3)
        flip_horizontal: Flip horizontally?
        flip_vertical: Flip vertically?

    Returns:
        Flipped data with the same shape.
    """

    if flip_horizontal:
        data=data[:,:,::-1]
    if flip_vertical:
        data=data[:,::-1,:]
    return data

def save_frame(path: Path, frame: xr.DataArray, ris: int, shot: int, time: float) -> Path:
    """Save single frame to image file.

    Output file name is f"{path}/RIS{ris}_{shot}_t={time}.png"

    Args:
        path: Where to save the image.
        frame: Array with data, shape (nx, ny, 3), values in the range <0, 1>.
        ris: Number of RIS camera, will be saved as a part of image file name.
        shot: Shot, will be saved as a part of image file name.
        time: Time in [ms]. Will be saved as a part of image file name

    Returns:
        Path to created file name.

    Raises:
        ??
    """
    filename=f"{path}/RIS{ris}_{shot}_t={time}.png"
    img = Image.fromarray((frame.data*255).astype('uint8')).convert('RGB')
    img.save(filename, format=None)
    
    return filename

def discharge_duration(shot: int, threshold: float = 1e4) -> Tuple[float, float]:
    """Find discharge durationS

    Args:
        shot: Queried shot.
        threshold: I_plasma must be higher than this threshold to say there is a plasma.

    Returns:
        Tuple[plasma_start, plasma_end]
    """

    # Note: the data are huge ==> caching them for future use would soon deplete our memory => cache=False
    s = cdbxr.Shot(shot, cache=False) #load the whole shot
    ipla = s[f"I_plasma:{shot}"] #load current
    plasma_time = ipla[abs(ipla)>threshold].time.data
    start = plasma_time[0]
    end = plasma_time[-1]

    return  start, end

def save_ris_images_to_folder(data: int, shot, path: Path, ris: int, use_discharge_duration=False):
    """Save all images from RIS camera in a given shot to a given folder.

    Saves only frames from times when there was a plasma.

    Args:
        shot: Shot.
        path: Output path. The image files will be saved to subfolder path / {shot}. The subfolder
            will be created if it does not exist.
        ris: Which camera to use? 1 or 2.
    Returns:
        filenames: np.array with all the names of saved images
    """
    filenames = np.array([]) #str array with all the names of saved images

    if use_discharge_duration:
        start, end = discharge_duration(shot, 1e4)
        dem_data=flip_image(demosaic(data)[1].sel(time=slice(start,end)))

    else:
        dem_data=flip_image(demosaic(data)[1])
    
    for i,frame in enumerate(dem_data):
        filename = save_frame(path=path,frame=frame,ris=ris, shot=shot, time=frame.time.data)
        filenames = np.append(filenames,filename)
        if i%50==0:
            print(f'{i}/{len(dem_data)} imgs are saved')
    return filenames


def heat_flux_datasets(scaled_Qdata, test_split, train_split, dropouts, target_fields,):
    
    # Drop desired features
    for col in scaled_Qdata.columns:
        for feature in dropouts:
            if (col[:len(feature)]==feature):
                print(f'Dropping {col}')
                scaled_Qdata = scaled_Qdata.drop([col], axis=1)

    # Create randomized data set.
    data_set = scaled_Qdata.sample(n=scaled_Qdata.shape[0])

    ### Split the data to test, training, and validation sets

    # Define offsets
    data_set_size = data_set.shape[0]

    # use given % of all data for testing
    test_set_size = int(data_set_size * test_split)

    # use given % of training data for validation
    train_set_size = int((data_set_size - test_set_size) * train_split)
    valid_set_size = (data_set_size - test_set_size) - train_set_size

    # Split data
    test_set = data_set.iloc[:test_set_size]
    train_set = data_set.iloc[test_set_size:test_set_size+train_set_size]
    validation_set = data_set.iloc[test_set_size+train_set_size:]

    # Keep ordered data for visualization (contains test, training, and validation sets)
    vis_set = scaled_Qdata # All data (not randomized) is kept for visualization of model

    # Split sets to features and targets subsets
    test_features, test_targets =\
        test_set.drop(target_fields, axis=1), test_set[target_fields]
    train_features, train_targets =\
        train_set.drop(target_fields, axis=1), train_set[target_fields]
    validation_features, validation_targets =\
        validation_set.drop(target_fields, axis=1), validation_set[target_fields]
    # Special object for visualization
    vis_features, vis_targets =\
        vis_set.drop(target_fields, axis=1), vis_set[target_fields]

    # Create pytorch tensor format data sets
    test_set =\
        TensorDataset(torch.tensor(test_features.values).float(),\
                      torch.tensor(test_targets[target_fields].values).float())
    train_set =\
        TensorDataset(torch.tensor(train_features.values).float(),\
                      torch.tensor(train_targets[target_fields].values).float())
    validation_set =\
        TensorDataset(torch.tensor(validation_features.values).float(),\
                      torch.tensor(validation_targets[target_fields].values).float())
    # Special data set for visualization (ordered)
    vis_set =\
        TensorDataset(torch.tensor(vis_features.values).float(),\
                      torch.tensor(vis_targets[target_fields].values).float())
    
    return test_set, train_set, validation_set, vis_set
