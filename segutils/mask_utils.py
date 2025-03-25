import numpy as np

def get_bounding_box(X, buffer=1, error_if_noncontiguous=False):
    """
    Calculates the bounding box of a ndarray
    """

    X_shape = X.shape
    x_shape = tuple([i+2 for i in X_shape])

    x = np.zeros(x_shape)
    x[1:-1,1:-1] = X

    mask = x == 0
    bbox = []
    all_axis = np.arange(x.ndim)

    for kdim in all_axis:
        
        nk_dim = np.delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = np.diff(mask_i)
        idx_i = np.nonzero(dmask_i)[0]
        
        if len(idx_i) != 2:
            if error_if_noncontiguous:
                raise ValueError('Algorithm failed, {} does not have 2 elements!'.format(idx_i))
            else:
                idx_i = np.array([ np.min(idx_i), np.max(idx_i) ])
        
        bound_lo = idx_i[0]-buffer
        bound_hi = idx_i[1]+buffer
        
        bound_lo = np.clip(bound_lo, 0, X_shape[kdim]-1)
        bound_hi = np.clip(bound_hi, 0, X_shape[kdim]-1)

        bbox.append(slice(bound_lo, bound_hi))
        
    return tuple(bbox)