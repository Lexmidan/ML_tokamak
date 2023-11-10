from typing import Tuple   # v moderním pythonu lze nezávazně specifikovat typ argumentů funkce
from cdb_extras import xarray_support as cdbxr   # načítání dat z databáze COMPASSu
from pyCDB import client
import numpy as np         # práce s numerickými poli
from pathlib import Path   # reprezentace cest v souborovém systému
import xarray as xr        # data s osami
from PIL import Image
from tqdm.auto import tqdm

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
    g=xr.DataArray((g1.data+g2.data)/2) #don't wanna lose any information. Green will be normalized anyway

    xar=xr.DataArray([r,g,b], dims=('color', 'time', 'x', 'y'), coords={'color': ['r','g','b'], 'time': data.coords['time'],\
                                                                         'x': np.arange(r.shape[1]), 'y':np.arange(r.shape[2])})
    axr=xar.transpose('time', 'x', 'y', 'color')

    # kazdy pixel ma 12 bitu (hodnoty 0..4096) => normalizujeme na rozsah 0..1
    axr=axr/(2**12-1)  #Nejde tu pouzit axr/=... "Cannot cast ufunc 'divide' output from dtype('float64') to dtype('uint16') with casting rule 'same_kind'"

    #Normalizace - stara
    #np_axr = axr.data.astype('float64') 
    #channel_min = np.min(np_axr, axis=(0, 1, 2))  # minimum value for each time and channel
    #channel_span = np.max(np_axr, axis=(0, 1, 2)) - channel_min  # span of values for each time and channel
    #channel_norm = np.where(channel_span > 0, channel_span, 1)  # use 1 when span is 0
    #axr = (axr - channel_min[None, None, :]) / channel_norm[None, None, :]
    #np_axr=axr.data


    return axr

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

    s = cdbxr.Shot(shot) #load the whole shot
    ipla = s["I_plasma"] #load current
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
    filenames = [] #str array with all the names of saved images

    if use_discharge_duration:
        start, end = discharge_duration(shot, 1e4)
        dem_data=flip_image(demosaic(data).sel(time=slice(start,end)))

    else:
        dem_data=flip_image(demosaic(data))
    for frame in tqdm(dem_data, total=len(dem_data)):    # automaticky se zobrazi a bude v prubehu cyklu updatovat progressbar
        filename = save_frame(path=path,frame=frame,ris=ris, shot=shot, time=frame.time.data)
        filenames.append(filename)

    return filenames

