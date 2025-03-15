from copy import deepcopy
import numpy as np
import pandas as pd
from skimage.measure import find_contours, regionprops_table, EllipseModel

import matplotlib.pyplot as plt
import seaborn as sns

def remove_low_intensity_objects(intensity_image, input_labels_image, intensity_threshold=0.3):
    """
    Remove objects from input_labels_image whose mean intensity in `intensity_image`
    falls below the background intensity multiplied by `intensity_threshold`

    Note that pixels with value 0 in `input_labels_image` are considered to be
    background pixels.
    """

    im = deepcopy(input_labels_image)
    bg_intensity = np.median(intensity_image[input_labels_image==0])
    
    unique_labels = np.unique(im)[1:]

    for i, ilabel in enumerate(unique_labels):
        if np.mean(intensity_image[im==ilabel]) < (bg_intensity*(1+intensity_threshold)):
            im[im==ilabel] = 0
            
    return im

def remove_non_ellipsoidal_objects(input_labels_image, plot_residual_hist=False):
    """
    Removes non-ellipsoidal objects from a labeled image.

    This function identifies and removes objects in the input labeled image 
    that do not approximate well to an ellipse. It fits an ellipse to each 
    labeled object and calculates the residuals. Objects with high residuals 
    are considered non-ellipsoidal and are removed from the image.

    Parameters:
    input_labels_image (numpy.ndarray): A labeled image where each object is 
    represented by a unique integer label.

    Returns:
    numpy.ndarray: A labeled image with non-ellipsoidal objects removed.

    Note:
    - The function uses a residual threshold of 0.1 to determine whether an 
    object is ellipsoidal. Objects with median residuals greater than this 
    threshold are removed.
    - The function also plots a histogram of scaled residuals for each object 
    if `plot_residual_hist` is set to True.
    - Pixels with value 0 in `input_labels_image` are considered to be
    background pixels.
    """
    
    im = deepcopy(input_labels_image)
    
    unique_labels = np.unique(im)[1:]
    
    if plot_residual_hist:
        fig, ax = plt.subplots(figsize=(2,1.5))
        cmap = sns.color_palette("tab10")
        
    for i, ilabel in enumerate(unique_labels):
        xy = find_contours(im==ilabel, level=0.5)[0]
        em = EllipseModel()
        em_result = em.estimate(xy)
        
        if em_result: # fit success
            xhat_yhat = em.predict_xy(np.arange(0,2*np.pi,0.01))
        
            em_scaled_residuals = em.residuals(xy) / np.prod(em.params[2:4])**0.5
        
            if plot_residual_hist:
                bins = np.arange(0,0.4,0.02)
                plt.hist(em_scaled_residuals, bins=bins, color=cmap[i], label=ilabel, histtype="step")
                plt.axvline(x=np.median(em_scaled_residuals), color=cmap[i])
    
        if (not em_result) or (np.median(em_scaled_residuals) > 0.1):
            im[im == ilabel] = 0
    
    if plot_residual_hist:
        plt.legend()
        ax.set_xlabel("residual / mean axis length")
        ax.set_ylabel("frequency")

    return im


## old functions

def merge_buds(maxz_img, labels, linewidth=10, septin_threshold=150, gaussian_amp_threshold=20):
    """
    Given an image `maxz_img` and a corresponding labeled image `labels`, merge labels correspon-
    ding to mother cell-bud pairs.

    Pairs of labels to consider joining are determined using a region adjacency graph (rag).
    
    The criterion for joining two differently-labeled regions into a single-label budding cell is
    that the line joining the centroid of the two regions must have a local maximum (correspondi-
    ng to a labeled septin ring) whose brightness relative to the surrounding non-septin-ring re-
    gion exceeds `gaussian_amp_threshold`
    """

    if len(np.unique(labels)) == 1:
        return labels

    new_labels = deepcopy(labels)

    edge_map = sobel(labels)
    rag = graph.rag_boundary(labels, edge_map)
    rag.remove_node(0)

    rprops = pd.DataFrame(regionprops_table(labels, properties=('label', "area", 'centroid', 'bbox'),))

    cells_considered = []
    for icell, jcell in list(rag.edges):

        if (icell in cells_considered) or (jcell in cells_considered):
            continue

        intensity_profile = profile_line(
            maxz_img,
            rprops.loc[rprops["label"] == icell, ["centroid-0", "centroid-1"]].values.flatten(),
            rprops.loc[rprops["label"] == jcell, ["centroid-0", "centroid-1"]].values.flatten(),
            linewidth=linewidth
        )

        def func(x, a, b, c, d):
            return a*np.exp(-((x-b)/c)**2) + d

        xdata = np.arange(len(intensity_profile))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(func, xdata, intensity_profile)
                lo_lim = popt[1] - 2*popt[2]
                hi_lim = popt[1] + 2*popt[2]

            profile_maxima = peak_local_max(intensity_profile, threshold_abs=septin_threshold).flatten()

            if (len(profile_maxima) > 0) and ((lo_lim > 0) and (hi_lim < np.max(xdata)) and (popt[0] > gaussian_amp_threshold)): # and ~np.any(np.isnan(pcov))):
                icell_area = rprops.loc[rprops["label"] == icell, "area"].values.flatten()
                jcell_area = rprops.loc[rprops["label"] == jcell, "area"].values.flatten()
                if icell_area > jcell_area:
                    new_labels[new_labels==jcell] = icell
                else:
                    new_labels[new_labels==icell] = jcell
                cells_considered.append(icell)
                cells_considered.append(jcell)
        except:
            continue

    return new_labels

def calculate_mito_shift(cyto_mask, mito_img, maxdxy=20, plot=False):
    """
    Given a cytoplasm mask and mitochondrial image, shift the mitochondrial image to minimize the amount of mitochondria outside the cytoplasmic mask.

    (Lack of) overlap is calculated manually over the range of shifts [-maxdxy, maxdxy] in x and y, and the shift which minimizes this is selected.

    This should be generally useful for registration of multiple channels where positive pixels in one are a strict subet of positive pixels in the other.

    Returns:
    mito_img_reg: shifted mito_img
    (dx, dy): x and y shifts in pixels, which can then be applied to other images or volumes
    """

    mito_blur = gaussian(mito_img, sigma=1)

    gridsearch_result = []

    for dx in np.arange(-maxdxy, maxdxy+1):
        for dy in np.arange(-maxdxy, maxdxy+1):
            mito_blur_masked = np.roll(mito_blur, (dx, dy), axis=(0,1))
            mito_blur_masked[cyto_mask] = 0
            mitosum_outside_cellmask = np.sum(mito_blur_masked)
            gridsearch_result.append([dx, dy, mitosum_outside_cellmask])

    dx, dy, msocm = tuple(np.array(gridsearch_result).T)

    dx_min = int(dx[np.argmin(msocm)])
    dy_min = int(dy[np.argmin(msocm)])

    mito_img_reg = np.roll(mito_img, (dx_min, dy_min), axis=(0,1))

    if dx_min > 0:
        mito_img_reg[:dx_min,:] = 0
    if dy_min > 0:
        mito_img_reg[:,:dy_min] = 0

    if dx_min < 0:
        mito_img_reg[dx_min:,:] = 0
    if dy_min < 0:
        mito_img_reg[:,dy_min:] = 0

    if plot:
        plt.scatter(dx, dy, c = msocm - np.min(msocm), marker="s")
        plt.scatter(dx_min, dy_min, c="crimson", marker="s")
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.gca().set_aspect(1)
        plt.show()

    return mito_img_reg, (dx_min, dy_min)

# viewer.add_labels(remove_non_ellipsoidal_objects(da_seg_curr_3d.data[2,:]), name="seg_corr")