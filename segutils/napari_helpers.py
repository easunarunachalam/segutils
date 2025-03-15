from copy import deepcopy
import numpy as np


def singlez_to_multiz(da_singlez, da_matchz):
    """
    Extend a 2D data array to match the depth of a 3D data array by replicating the 2D data across all z-slices.

    This function takes a 2D data array and a 3D data array and creates a new 3D data array by 
    repeating the 2D array along the z-axis. The number of repetitions matches the number of 
    z-slices in the 3D data array. The resulting 3D array retains the metadata structure of the 
    input 3D data array.

    Parameters:
    -----------
    da_singlez : DataArray
        A 2D data array that needs to be extended to a 3D array. This is typically a single 
        slice or a label map to be applied uniformly across all z-slices.
    
    da_matchz : DataArray
        A 3D data array whose z-dimension length determines the number of slices to replicate 
        the 2D array across. This array provides the metadata structure (including dimensions 
        and coordinates) for the output array.

    Returns:
    --------
    da_multiz : DataArray
        A new 3D data array where the 2D input data has been replicated across all z-slices, 
        matching the z-dimension of `da_matchz`. The output retains the structure and metadata 
        of `da_matchz`, including dimension names and coordinates.
    """
    
    nz = len(da_matchz.Z)
    
    data = np.repeat(da_singlez.data[np.newaxis,:,:], nz, axis=0)
    
    da_multiz = deepcopy(da_matchz)
    da_multiz.data = data
    
    return da_multiz