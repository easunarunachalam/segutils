import numpy as np

def get_bounding_box(X, buffer=1):
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
            print(idx_i)
            raise ValueError('Algorithm failed, {} does not have 2 elements!'.format(idx_i))
        bbox.append(slice(idx_i[0]-buffer, idx_i[1]+buffer))
        
    return tuple(bbox)