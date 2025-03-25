import numpy as np

def stack_different_shape_arrays(arrays, axis=0, buffer_width=1, buffer_value=0):
    """
    Stacks multiple NumPy arrays with different dimensions along `axis`,
    inserting a buffer of specified width and value between them. Works for arrays of arbitrary dimensions.
    
    Parameters:
    arrays (list of np.ndarray): List of NumPy arrays to be stacked.
    axis (int): Axis along which to stack the arrays.
    buffer_width (int): Width of the buffer to insert between arrays.
    buffer_value (numeric): Value to fill the buffer with.
    
    Returns:
    np.ndarray: The stacked array with buffers.
    """
    if not arrays:
        raise ValueError("Input list of arrays is empty.")
    
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise ValueError("All elements of the input list must be NumPy arrays.")
    
    # Determine the maximum shape along each axis
    array_shapes = np.array([arr.shape for arr in arrays])
    # print(array_shapes)
    max_shape = array_shapes.max(axis=0)
    
    # Pad arrays to match max shape along all non-stacking axes
    def pad_to_max_shape(arr, max_shape):
        # pad_width = [(0, max_dim - arr_dim) for arr_dim, max_dim in zip(arr.shape, max_shape)]
        pad_width = []
        for idim, (arr_dim, max_dim) in enumerate(zip(arr.shape, max_shape)):
            if idim != axis:
                pad_width.append( (0, max_dim - arr_dim) )
            else:
                pad_width.append((0,0))
        # print("pad width", pad_width)
        return np.pad(arr, pad_width, constant_values=buffer_value)
    
    
    padded_arrays = [pad_to_max_shape(arr, max_shape) for arr in arrays]

    for i_arr in range(len(padded_arrays)):
        while axis > (len(padded_arrays[i_arr].shape)-1):
            padded_arrays[i_arr] = np.expand_dims(padded_arrays[i_arr], axis=len(padded_arrays[i_arr].shape))

    # Determine the maximum shape along each axis
    array_shapes = np.array([arr.shape for arr in padded_arrays])
    # print(array_shapes)
    max_shape = array_shapes.max(axis=0)

    # if buffer_width == 0:
    #     # if no buffer, directly stack padded arrays
    #     return np.stack(padded_arrays, axis=axis)
    # else:

    # if a buffer is required:
    # Create buffer array with appropriate shape
    buffer_shape = list(max_shape)
    buffer_shape[axis] = buffer_width
    buffer = np.full(buffer_shape, buffer_value, dtype=arrays[0].dtype)
    
    # Interleave arrays with buffer
    stacked = []
    for i, arr in enumerate(padded_arrays):
        stacked.append(arr)
        if i < len(padded_arrays) - 1:
            stacked.append(buffer)
    
    return np.concatenate(stacked, axis=axis)



# def stack_different_shape_arrays(arrays, axis=0, buffer_width=1, buffer_value=0):
#     """
#     Stacks multiple 2D NumPy arrays with different dimensions along `stack_axis`,
#     inserting a buffer of specified width and value between them. Also allows stacking along a new third axis.
    
#     Parameters:
#     arrays (list of np.ndarray): List of 2D numpy arrays to be stacked.
#     stack_axis (int): Axis along which to stack the arrays (0 for vertical, 1 for horizontal, 2 for depth).
#     buffer_width (int): Width of the buffer to insert between arrays.
#     buffer_value (numeric): Value to fill the buffer with.
    
#     Returns:
#     np.ndarray: The stacked array with buffers.
#     """
#     if not arrays:
#         raise ValueError("Input list of arrays is empty.")
    
#     if axis not in [0, 1, 2]:
#         raise ValueError("stack_axis must be 0 (vertical), 1 (horizontal), or 2 (depth).")
    
#     # Determine the max shape along the non-stacking axes
#     max_height = max(arr.shape[0] for arr in arrays)
#     max_width = max(arr.shape[1] for arr in arrays)
    
#     padded_arrays = [
#         np.pad(arr, ((0, max_height - arr.shape[0]), (0, max_width - arr.shape[1])), constant_values=buffer_value)
#         for arr in arrays
#     ]

#     print("padded shape", padded_arrays[0].shape, len(padded_arrays), len(arrays))
    
#     if axis == 0:
#         buffer_shape = (buffer_width, max_width)
#     elif axis == 1:
#         buffer_shape = (max_height, buffer_width)
#     else:
#         buffer_shape = (max_height, max_width, buffer_width)
#         padded_arrays = [arr[:, :, np.newaxis] for arr in padded_arrays]  # Expand arrays to 3D
    
#     print("buffer shape", buffer_shape)

#     # Create buffer array
#     buffer = np.full(buffer_shape, buffer_value, dtype=arrays[0].dtype)
    
#     # Interleave arrays with buffer
#     stacked = []
#     for i, arr in enumerate(padded_arrays):
#         stacked.append(arr)
#         if i < len(padded_arrays) - 1:
#             stacked.append(buffer)
    
#     return np.concatenate(stacked, axis=axis)

# def stack_different_shape_arrays(arrays, stack_axis=0, buffer_width=1, buffer_value=0):
#     """
#     Stacks multiple 2D Numpy arrays with different dimensions along `stack_axis`,
#     inserting a buffer of specified width and value between them.
    
#     Parameters:
#     arrays (list of np.ndarray): List of 2D numpy arrays to be stacked.
#     stack_axis (int): Axis along which to stack the arrays (0 for vertical, 1 for horizontal).
#     buffer_width (int): Width of the buffer to insert between arrays.
#     buffer_value (numeric): Value to fill the buffer with.
    
#     Returns:
#     np.ndarray: The stacked array with buffers.
#     """
#     if not arrays:
#         raise ValueError("Input list of arrays is empty.")
    
#     if stack_axis not in [0, 1]:
#         raise ValueError("stack_axis must be 0 (vertical) or 1 (horizontal).")
    
#     # Determine the max shape along the non-stacking axis
#     if stack_axis == 0:
#         max_width = max(arr.shape[1] for arr in arrays)
#         padded_arrays = [np.pad(arr, ((0, 0), (0, max_width - arr.shape[1])), constant_values=buffer_value) for arr in arrays]
#         buffer_shape = (buffer_width, max_width)
#     el:
#         max_height = max(arr.shape[0] for arr in arrays)
#         padded_arrays = [np.pad(arr, ((0, max_height - arr.shape[0]), (0, 0)), constant_values=buffer_value) for arr in arrays]
#         buffer_shape = (max_height, buffer_width)
    
#     # Create buffer array
#     buffer = np.full(buffer_shape, buffer_value, dtype=arrays[0].dtype)
    
#     # Interleave arrays with buffer
#     stacked = []
#     for i, arr in enumerate(padded_arrays):
#         stacked.append(arr)
#         if i < len(padded_arrays) - 1:
#             stacked.append(buffer)
    
#     return np.concatenate(stacked, axis=stack_axis)