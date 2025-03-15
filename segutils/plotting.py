import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours


def shuffle_mask_labels(mask_image, label_mapping=None, exclude_0=True):
    """
    Shuffle the labels in a mask image, optionally excluding label '0' from the shuffle.

    This function takes a labeled mask image, identifies the unique labels, and randomly shuffles them. 
    It then applies the shuffled labels back to the mask image. If `exclude_0` is set to `True`, the label '0' 
    (often used for background) will not be shuffled and will remain unchanged.

    This is helpful e.g. when you want to show a segmented image where the adjacent masks have consecutive indices,
    and you want to be able to distinguish them easily. For example,
    >>> mask_image = shuffle_mask_labels(masks)
    >>> mask_image[mask_image == 0] = -1
    >>> cmap = plt.get_cmap("tab20", np.max(mask_image) + 1)  # Get a colormap with enough colors
    >>> cmap.set_under("white")
    >>> ax.matshow(mask_image, cmap=cmap, vmin=0)

    Args:
        mask_image (numpy.ndarray): The input mask image, where each pixel value corresponds to a label.
        exclude_0 (bool, optional): If True, label '0' will be excluded from shuffling and remain unchanged.
                                     Defaults to True.

    Returns:
        numpy.ndarray: A new mask image with shuffled labels.
    
    Example:
        >>> mask_image = np.array([[1, 2, 2], [1, 0, 3], [3, 3, 1]])
        >>> shuffled_mask = shuffle_mask_labels(mask_image, exclude_0=True)
    """

    # If not already provided, create a label mapping from old labels to new ones
    if label_mapping is None:
        ## Step 1: Find unique mask labels
        unique_labels = np.unique(mask_image)

        if exclude_0:
            unique_labels = unique_labels[unique_labels != 0]
        
        ## Step 2: Shuffle the labels
        shuffled_labels = unique_labels.copy()
        np.random.shuffle(shuffled_labels)
        
        ## Step 3: Create a mapping from original labels to shuffled labels
        label_mapping = dict(zip(unique_labels, shuffled_labels))
        if exclude_0:
            label_mapping = label_mapping | {0: 0}
    

    # apply the shuffled labels to the mask image
    shuffled_mask_image = np.vectorize(label_mapping.get)(mask_image)

    return shuffled_mask_image, label_mapping

def show_labels_image(ax, masks, label_mapping=None, cmap=None, bg_color="white", cbar=False):

    shuf_masks, label_mapping = shuffle_mask_labels(masks, label_mapping=label_mapping)
    shuf_masks[shuf_masks == 0] = -1

    if cmap is None:
        cmap = plt.get_cmap("tab20", np.max(shuf_masks) + 1)  # Get a colormap with enough colors
        cmap.set_under(bg_color)

    p = ax.matshow(shuf_masks.T, cmap=cmap, vmin=0)

    if cbar:
        plt.colorbar(p)

    return label_mapping, cmap

def draw_contour_binary(ax, binary_image, color="crimson", lw=2):
    """
    Draws contours of a binary image given an axis.

    This function finds contours in a binary image (a mask where pixels are either 0 or 1) 
    and plots them on a provided `matplotlib` axis. The contours are drawn with the specified color.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which the contours will be drawn.

    binary_image : ndarray
        A binary image (2D numpy array) where non-zero values represent the object and 
        zero values represent the background.

    color : str, optional, default "crimson"
        The color of the contours to be drawn.

    lw : float, optional, default 2
        The linewidth of the contours to be drawn.

    Returns:
    --------
    None
        This function modifies the `ax` object directly by drawing the contours.

    Notes:
    ------
    - This function uses `find_contours` from `skimage.measure` to detect contours at a 
      specified level (0.5, which works well for binary masks).

    Example:
    --------
    # Example usage with a binary mask and matplotlib axis
    fig, ax = plt.subplots()
    binary_image = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]])
    draw_contour_binary(ax, binary_image, color="blue")
    plt.show()
    """

    contours = find_contours(binary_image, level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=lw, color=color)