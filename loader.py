import numpy as np
import spectral.io.envi as envi
import netCDF4 as nc
from pathlib import Path
import utilities as util

def load_envi(precision, path, x_start=0, x_end=0, y_start=0, y_end=0):
    """
    This function returns a 2d np.array where of size (bands,pixels) where pixels are stacked

    Parameters:

    precision: np.dtype
    filepath: string

    returns
    np.array: stacked datacube representation
    """
    header_string = path + ".hdr"

    img = envi.open(header_string,path)
    assert (x_end >= x_start) and (y_end >= y_start), "load_envi(): Coordinates not valid"
    if x_start + x_end == 0:
        x_end = img.shape[0]
    if y_start + y_end == 0:
        y_end = img.shape[1]
    arr = img.read_subimage(rows=range(x_start, x_end), cols=range(y_start, y_end)).astype(precision)
    return arr

def load_RGB(Data, R_band=44, G_band=18, B_band=6):
    assert len(Data.shape) == 3, "Data is not 3 dimensional"
    R = Data[:, :, R_band]
    G = Data[:, :, G_band]
    B = Data[:, :, B_band]

    rgb_image = np.stack((R, G, B), axis=-1)

    return rgb_image

def load_l1b_shape(string: Path): #TODO doc
    assert string.name.endswith('l1b.nc'), "File is not l1b processed"
    with nc.Dataset(string, "r", format="NETCDF4") as rootgrp:
        shape = rootgrp.groups["products"].variables["Lt"][:].shape #lines, samples, bands
        return shape

def load_l1b_cube(string: Path): #TODO doc
    assert string.name.endswith('l1b.nc'), "File is not l1b processed"
    with nc.Dataset(string, "r", format="NETCDF4") as f:
        group = f.groups["products"]
        data = np.array(group.variables["Lt"][:,:,3:], dtype=np.float64)
        return data