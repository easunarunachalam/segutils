from copy import deepcopy
import imageio as iio
import numpy as np
from pathlib import Path
from tqdm.autonotebook import trange
import xarray as xr

C_AXIS = 0
Z_AXIS = 1
Y_AXIS = 2
X_AXIS = 3
axis_order = list("CZYX")


def reshape_consistent(fns, nc, nz):

    for ifn in trange(len(fns)):

        im = iio.volread(fns[ifn])
        curr_axis_order = deepcopy(axis_order)

        c_axis = np.where([idim==nc for idim in im.shape])[0][0]
        curr_axis_order[c_axis] = "C"

        z_axis = np.where([idim==nz for idim in im.shape])[0][0]
        curr_axis_order[z_axis] = "Z"

        print(curr_axis_order)

        orig_path   = Path(fns[ifn])
        stem_wo_ome = Path(Path(fns[ifn]).stem).stem
        new_path    = orig_path.with_stem(stem_wo_ome + "_constdim").with_suffix(".nc")

        da = xr.DataArray(
            data=im,
            dims=[i for i in curr_axis_order],
        ).transpose(*axis_order)

        da.to_netcdf(new_path, mode="w")
        
        del da

# def reshape_consistent(fns, nc, nz):

#     for ifn in trange(len(fns)):

#         im = iio.volread(fns[ifn])
#         print(i.shape, )

#         c_axis = np.where([idim==nc for idim in im.shape])[0][0]
#         im = np.moveaxis(im, c_axis, C_AXIS)

#         z_axis = np.where([idim==nz for idim in im.shape])[0][0]
#         im = np.moveaxis(im, z_axis, Z_AXIS)

#         orig_path   = Path(fns[ifn])
#         stem_wo_ome = Path(Path(fns[ifn]).stem).stem
#         new_path    = orig_path.with_stem(stem_wo_ome + "_constdim").with_suffix(".nc")

#         da = xr.DataArray(
#             data=im,
#             dims=[i for i in axis_order],
#         )

#         da.to_netcdf(new_path, mode="w")
        
#         del da

def create_all_maxz_projs(fns):

    for ifn in trange(len(fns)):

        da = xr.load_dataarray(fns[ifn])
        print(da.dims, da.shape)

        da_maxz = da.max("Z")

        orig_path = Path(fns[ifn])
        maxz_path = orig_path.with_stem(orig_path.stem.replace("_constdim", "_maxz"))

        da_maxz.to_netcdf(maxz_path, mode="w")
        del da
        del da_maxz
