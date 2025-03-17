from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.signal import butter, filtfilt, find_peaks
from skimage.measure import profile, profile_line
from tqdm.notebook import tqdm, trange

import matplotlib.pyplot as plt

import sys
sys.path.append("D:/Dropbox/code/pyutils")
sys.path.append("D:/Dropbox/code/")
from pyutils import ODR

def pairwise_distances_bw_masks(masks, close_thresh=2):
    """
    Computes pairwise distances between labeled masks in a given segmentation mask.

    This function calculates the Euclidean distance transform for each labeled 
    region in the mask and determines pairwise distances between all labeled 
    regions. It also identifies pixels that are within a specified threshold 
    distance (`close_thresh`) from another region.

    Parameters:
    -----------
    masks : np.ndarray
        A 2D array where each unique nonzero integer represents a labeled 
        region, and zero represents the background.
    close_thresh : int, optional
        The maximum distance (in pixels) to consider two regions as "close." 
        Default is 2.

    Returns:
    --------
    df_pairwise_dists : pd.DataFrame
        A DataFrame containing pairwise distance metrics with columns:
        - "label_i": Label of the first region.
        - "label_j": Label of the second region.
        - "mean_distance": Mean distance between pixels of the two regions.
        - "num_close_px": Number of pixels in both regions within `close_thresh` 
          distance of each other.
    close_masks : list of np.ndarray
        A list of binary masks, where each mask highlights the pixels that 
        are within `close_thresh` distance between a pair of regions.

    Notes:
    ------
    - The function excludes the background (label 0) from analysis.
    - Uses `scipy.ndimage.distance_transform_edt` to compute distance transforms.
    - Iterates over all unique label pairs to compute distances.
    """


    # get list of cell labels
    labels = np.unique(masks)
    labels = labels[labels != 0] # exclude background (0)
    n_labels = len(labels)
    
    # calculate distance transforms for inverse of each mask
    dist_all_lbl = []
    for i in trange(n_labels, desc="distance transforms", leave=False):
        lbl_i = labels[i]
        
        # mask for label i
        mask_lbl_i = (masks == lbl_i)
        
        # distance from edge of label i
        dist_lbl_i = distance_transform_edt(mask_lbl_i==0)
    
        dist_all_lbl.append(dist_lbl_i)
    
    df_pairwise_dists = pd.DataFrame(columns=["label_i", "label_j", "mean_distance", "num_close_px"])

    close_masks = []
    
    pbar = tqdm(total=(n_labels*(n_labels-1))/2, desc="pairwise distances", leave=False)

    for i in range(n_labels):
        for j in range(i):
            
            lbl_j = labels[j]
            lbl_i = labels[i]
    
            # mask for labels i, j
            mask_lbl_i = (masks == lbl_i)
            mask_lbl_j = (masks == lbl_j)
            
            # distance from edge of labels i, j
            dist_lbl_i = dist_all_lbl[i]
            dist_lbl_j = dist_all_lbl[j]
    
            # distance of j px from edge of label i
            j_dists_from_i = dist_all_lbl[i][mask_lbl_j]
    
            # distance of i px from edge of label j
            i_dists_from_j = dist_all_lbl[j][mask_lbl_i]
    
            
            close_mask_ji = np.multiply(dist_lbl_i <= close_thresh, masks == lbl_j)
            close_mask_ij = np.multiply(dist_lbl_j <= close_thresh, masks == lbl_i)
            close_mask = close_mask_ji + close_mask_ij
            close_masks.append(close_mask)
    
            df_pairwise_dists.loc[len(df_pairwise_dists)] = {
                "label_i": lbl_i,
                "label_j": lbl_j,
                "mean_distance": np.mean(np.hstack((j_dists_from_i, i_dists_from_j))),
                "num_close_px": np.sum(close_mask_ji) + np.sum(close_mask_ij)
            }

            pbar.update(1)
    
    return df_pairwise_dists, close_masks


def low_pass_filter(input_array, cutoff_freq, sample_rate):
    """
    Apply a low-pass Butterworth filter to the input array.
    
    Parameters:
    - input_array: A 1D numpy array.
    - cutoff_freq: The cutoff frequency of the low-pass filter.
    - sample_rate: The sample rate (Hz) of the input signal.
    
    Returns:
    - A 1D numpy array after low-pass filtering.
    """
    # Normalizing the cutoff frequency
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the low-pass Butterworth filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the input data
    filtered_array = filtfilt(b, a, input_array)
    
    return filtered_array

def subtract_low_pass(input_array, cutoff_freq, sample_rate):
    """
    Apply a low-pass Butterworth filter to the input array and subtract the 
    filtered version from the original array to highlight high-frequency components.
    
    Parameters:
    - input_array: A 1D numpy array.
    - cutoff_freq: The cutoff frequency of the low-pass filter.
    - sample_rate: The sample rate (Hz) of the input signal.
    
    Returns:
    - A 1D numpy array which is the difference between the original and filtered arrays.
    """
    # Normalizing the cutoff frequency
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    # Design the low-pass Butterworth filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the input data
    filtered_array = filtfilt(b, a, input_array)
    
    # Subtract the filtered array from the original
    result = input_array - filtered_array
    
    return result

def merge_close_masks(orig_masks, intensity_image, close_thresh=2):
    """
    Merges segmented regions in a mask based on intensity profile analysis along their shared border.
    
    This process identifies pairs of labeled regions that are close together, 
    computes an intensity profile along their shared boundary, and determines 
    whether to merge the regions based on the presence of peaks and troughs 
    in the profile.
    
    Steps:
    1. Identify pairs of regions that have more than 2 close pixels.
    2. Fit a line to the boundary pixels of each close pair.
    3. Compute the intensity profile perpendicular to the fitted line.
    4. Apply a low-pass filter to the intensity profile.
    5. Detect peaks and troughs in the filtered intensity profile.
    6. Determine whether to merge the regions based on the peak and trough locations.
    7. Merge the smaller region into the larger one if merging criteria are met.
    
    Parameters:
    -----------
    df_pairwise_dists : pd.DataFrame
        DataFrame containing pairwise distance metrics between labeled regions.
    close_masks : list of np.ndarray
        List of binary masks indicating pixels that are close between region pairs.
    orig_masks : np.ndarray
        The original labeled mask where unique integers represent different regions.
    intensity_image : ndarray
        Image data representing intensity values to analyze along region boundaries.
    
    Returns:
    --------
    merged_masks : np.ndarray
        A modified version of `orig_masks` where certain adjacent regions have 
        been merged based on intensity profile analysis.
    
    Notes:
    ------
    - Uses orthogonal distance regression (ODR) to fit the boundary line.
    - The intensity profile is computed using `profile_line`.
    - Merging is based on the proximity of peaks relative to the midpoint of the profile.
    - The smaller region is merged into the larger one if merging conditions are met.
    """
    

    df_pairwise_dists, close_masks = pairwise_distances_bw_masks(orig_masks, close_thresh=close_thresh)
    
    df_close_pairs = df_pairwise_dists.loc[df_pairwise_dists["num_close_px"] > 2]
    
    merged_masks = deepcopy(orig_masks)
    
    for i in df_close_pairs.index:

        lbl_i = df_close_pairs.loc[i,"label_i"]
        lbl_j = df_close_pairs.loc[i,"label_j"]
    
        close_mask = close_masks[i]
    
        
        # identify pixels along the border between cells
        # fit a line to this set of pixels, and use this calculate the intensity profile perpendicular to this line
        y, x = tuple(np.argwhere(close_mask).T.astype(float))
        X = x.reshape((-1,1))
        
        # lr = LinearRegression(fit_intercept=True).fit(X, y)
        # lr_m, lr_b = lr.coef_[0], lr.intercept_
    
        s = 1
        (lr_m, lr_b), yhat_closepx = ODR(x, y, np.full_like(x,s), np.full_like(y,s), x_predict=x, return_stats=True)
        
        lr_perp_m = -1. / lr_m
        
        dx = np.max(x) - np.min(x)
        dy = np.max(y) - np.min(y)
        profile_width = int(np.sqrt(dx**2 + dy**2) / 1)
        
        x_mid, y_mid = np.median(x), np.median(y)
        
        
        dl = 100
        dx = dl/np.sqrt(1+lr_perp_m**2)
        dy = lr_perp_m*dx
        
        x_start, x_end = int(x_mid-dx), int(x_mid+dx)
        y_start, y_end = int(y_mid-dy), int(y_mid+dy)
        start_coord, end_coord = (y_start, x_start), (y_end, x_end)
        
        mask_lbl_i = (orig_masks == lbl_i)
        mask_lbl_j = (orig_masks == lbl_j)
        mask_lbl_ij = (mask_lbl_i + mask_lbl_j) > 0
        intensity_image_ij_only = np.multiply( intensity_image.astype(float), mask_lbl_ij.astype(float) )
        
        sum_intensity_profile = profile_line(intensity_image_ij_only, start_coord, end_coord, linewidth=profile_width, mode="constant", cval=0)
        sum_npx_profile = profile_line(intensity_image_ij_only > 0, start_coord, end_coord, linewidth=profile_width, mode="constant", cval=0)

        valid_idx = sum_npx_profile != 0
        avg_intensity_profile = sum_intensity_profile[valid_idx] / sum_npx_profile[valid_idx]

        try:
            avg_intensity_profile_lowpass = low_pass_filter(avg_intensity_profile,0.3,1)
        except:
            continue
        
        line_profile_coords = profile._line_profile_coordinates(start_coord, end_coord).squeeze()
        y_scan_coords, x_scan_coords = line_profile_coords[:,valid_idx]
        idx_closest_to_mid = np.argmin( (x_scan_coords-x_mid)**2 + (y_scan_coords-y_mid)**2 )
        peaks = find_peaks(avg_intensity_profile_lowpass, prominence=5)[0]
        troughs = find_peaks(-avg_intensity_profile_lowpass, prominence=5)[0]
    
        merge = False
        
        if len(peaks) > 0:
            
            thresh_peak_close_to_mid = 10
            peak_close_to_mid = np.any(np.abs(peaks - idx_closest_to_mid)) < thresh_peak_close_to_mid
    
            if len(troughs) > 0:
                
                peak_closer_than_trough = np.min(np.abs(peaks - idx_closest_to_mid)) < np.min(np.abs(troughs - idx_closest_to_mid))
                
                if peak_close_to_mid and peak_closer_than_trough:
                    merge = True
    
            else:
                if peak_close_to_mid:
                    merge = True
    
        if merge:
            npx_i = np.sum(orig_masks==lbl_i)
            npx_j = np.sum(orig_masks==lbl_j)
    
            if npx_i >= npx_j:
                merged_masks[orig_masks==lbl_j] = lbl_i
            else:
                merged_masks[orig_masks==lbl_i] = lbl_j

    return merged_masks