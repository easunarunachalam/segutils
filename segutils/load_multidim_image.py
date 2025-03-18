import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from contextlib import redirect_stdout
import dask.array as daskarray
import io
import xarray as xr
from IPython.display import display

def load_multi_zpitch_image(img_fns, channel_info=None, dtype=np.uint16, squeeze=True, compute=True, addl_metadata={}):
    """
    Load multidimensional images (names in the list `img_fns`) even if ImageJ
    metadata is unavailable, such as when a BeanShell script is used to
    control acquisitions.

    """

    assert isinstance(channel_info, dict)

    channel_names = channel_info.keys()
    n_channels = len(channel_names)
    n_slices_each_channel = [channel_info[key] for key in channel_names]

    need_to_manually_reshape = False
    
    ims = []

    # Create a dummy stream to capture output
    dummy_stdout = io.StringIO()

    for i in range(len(img_fns)):
        
        # silence errors/warnings associated with image import
        # then import the image
        with redirect_stdout(dummy_stdout):
            print("redirected")
            im = AICSImage(img_fns[i])

        if (im.shape[0] == 1) and (im.shape[1] != n_channels):
            need_to_manually_reshape = True
            ims.append(im)
        else:
            im = im.xarray_dask_data.astype(dtype)
            break

    if need_to_manually_reshape:

        im = daskarray.concatenate([im.dask_data for im in ims], axis=2)

        cum_n_slices = [0,] + list(np.cumsum(n_slices_each_channel))
        cum_n_slices = np.array(cum_n_slices, dtype=int)
        n_slices_per_timepoint = cum_n_slices[-1]

        target_shape = list(im.shape)
        target_shape[0] = target_shape[2] // n_slices_per_timepoint
        target_shape[2] = n_slices_per_timepoint
        
        im = im.reshape(target_shape, limit="128 MiB")
        im = AICSImage(im).xarray_dask_data.astype(dtype)

        # split image into separate arrays for each channel
        im_sepchannels = []
        for ichannel in range(len(channel_names)):
            im_ichannel = im.sel(Z=slice(cum_n_slices[ichannel], cum_n_slices[ichannel+1]))
            im_sepchannels.append(im_ichannel)

    else:

        # split image into separate arrays for each channel
        im_sepchannels = []
        max_zslices = np.max(n_slices_each_channel)
        for ichannel in range(len(channel_names)):
            if n_slices_each_channel[ichannel] < max_zslices:
                if n_slices_each_channel[ichannel] == 1:
                    use_z = slice(None,None,None)
                else:
                    interval = (max_zslices-1)//(n_slices_each_channel[ichannel]-1)
                    use_z = slice(None,None,interval)
            else:
                use_z = slice(None,None,None)
                
            im_sepchannels.append(im.isel(C=ichannel, Z=use_z))

    # add additional metadata
    for ichannel in range(len(im_sepchannels)):
        im_sepchannels[ichannel] = im_sepchannels[ichannel].assign_attrs(addl_metadata)

    # # add z coordinate values
    # for ichannel in range(len(im_sepchannels)):
    #     if len(channels[ichannel]) == 3:
    #         im_sepchannels[ichannel]["Z"] = np.arange()

    if squeeze:
        for ichannel in range(len(im_sepchannels)):
            im_sepchannels[ichannel] = im_sepchannels[ichannel].squeeze()    

    if compute:
        for ichannel in range(len(im_sepchannels)):
            im_sepchannels[ichannel] = im_sepchannels[ichannel].compute()

    return im_sepchannels


def clean_attrs(data_array):
    """
    Remove "unprocessed" attribute from MicroManager-generated data.
    """
    # Remove the "unprocessed" attribute
    if "unprocessed" in data_array.attrs:
        del data_array.attrs["unprocessed"]

def save_sep_channels(imgs, img_fns):
    """
    Save a list of xarray DataArrays to separate NetCDF files.

    Parameters
    ----------
    imgs : list of xarray.DataArray
        List of DataArray objects to be saved.
    img_fns : list of str
        List of filenames to save each DataArray. The length of img_fns should
        be the same as the length of imgs.

    Raises
    ------
    ValueError
        If the length of imgs and img_fns are not the same.
    """
    # check if the lengths of imgs and img_fns are the same
    if len(imgs) != len(img_fns):
        raise ValueError("The length of imgs and img_fns must be the same.")
    
    # iterate over each DataArray and its corresponding filename
    for img, fn in zip(imgs, img_fns):

        # clean attributes
        clean_attrs(img)

        # save the DataArray to a NetCDF file
        fn = Path(fn).with_suffix(".nc")
        img.compute().to_netcdf(fn, format="NETCDF4", engine="h5netcdf")