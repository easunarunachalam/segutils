from copy import deepcopy
from functools import partial
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
from pathlib import Path

from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

from sklearn.ensemble import RandomForestClassifier
from skimage.feature import multiscale_basic_features, peak_local_max
from skimage.filters import gaussian, sobel, threshold_local, threshold_otsu
# from skimage.future import graph
from skimage import graph
from skimage.measure import find_contours, label, profile_line, regionprops_table
from skimage.morphology import binary_erosion, disk, remove_small_objects
from skimage.segmentation import watershed

from tqdm.notebook import tqdm, trange

import warnings

def mask_to_labels(image, footprint_size=5):
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((footprint_size, footprint_size)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    return labels

def fit_single_ellipse(im):
    edges = canny(im, sigma=3.0)
    result = hough_ellipse(edges, accuracy=2, threshold=5, min_size=15, max_size=80)
    result.sort(order='accumulator')

    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    return edges, ((yc, xc, a, b), orientation)

def cell_ellipsoidal_volume(im):
    labeled_mask = mask_to_labels((im>0))

    V_ellipsoids = []
    for iregion in np.unique(labeled_mask)[1:]:
        iregion_mask = labeled_mask == iregion
        ellipse_edges, ellipse_geom = fit_single_ellipse(iregion_mask)
        (yc, xc, a, b), orientation = ellipse_geom
        V_ellipsoids.append( (4/3)*a*b*min(a,b) )
    V_total = np.sum(V_ellipsoids)

    return V_total

def cell_mito_mask_union(labeled_cell_mask, labeled_mito_mask, cell_mito_overlap_threshold=10, cell_cell_overlap_threshold=20):

    new_mask = np.zeros_like(labeled_cell_mask)

    verbose=False
    icell_idxs = np.unique(labeled_cell_mask)[1:]

    for icell in icell_idxs:
        # if icell == 26 or icell == 21:
        #     verbose=True
        #     print(icell)
        # else:
        #     verbose=False
        icell_mask = labeled_cell_mask == icell
        icell_mito_mask_union = deepcopy(icell_mask)

        for imito in np.unique(labeled_mito_mask[icell_mask])[1:]:
            imito_mask = labeled_mito_mask == imito
            if np.sum(icell_mask[imito_mask]) > cell_mito_overlap_threshold:
                icell_mito_mask_union += imito_mask

        overlap_with_other_cell = False

        new_label = icell
        icell_new_mask = icell_mito_mask_union > 0
        jcell_idxs = np.unique(new_mask)[1:]
        for jcell in jcell_idxs:
            jcell_new_mask = new_mask == jcell
            # if (icell == 26 or jcell == 21) or (jcell == 26 or icell == 21):
            #     print(icell, jcell, np.sum(icell_new_mask[jcell_new_mask]))
            if np.sum(icell_new_mask[jcell_new_mask]) > cell_cell_overlap_threshold:
                overlap_with_other_cell = True

                icell_npx = np.count_nonzero(icell_new_mask)
                jcell_npx = np.count_nonzero(jcell_new_mask)

                if jcell_npx > icell_npx:
                    new_mask[icell_new_mask] = jcell
                    new_mask[jcell_new_mask] = jcell
                else:
                    new_mask[icell_new_mask] = icell
                    new_mask[jcell_new_mask] = icell
        # if verbose:
        #     plt.imshow(icell_mito_mask_union)
        #     plt.show()

        if not overlap_with_other_cell:
            new_mask[icell_mito_mask_union] = icell

    return new_mask

def mito_mask(mito_im, block_size=35, mito_im_blur_sigma=1, mito_im_blur_thresh=110, min_size=20, erosion_disk_radius=1, offset=5):

    mito_im_blur = gaussian(mito_im, sigma=mito_im_blur_sigma, preserve_range=True)

    # seed step
    local_thresh = threshold_local(mito_im_blur, block_size, offset=0)
    binary_local = mito_im_blur > local_thresh
    binary_nobg = binary_local * (mito_im_blur >= mito_im_blur_thresh)
    binary_nosmall = remove_small_objects(binary_nobg, min_size=min_size)
    binary_seeds = binary_erosion(binary_nosmall, footprint=disk(erosion_disk_radius))
    labeled_mito_seeds = label(binary_seeds)

    # watershed step
    local_thresh = threshold_local(mito_im_blur, block_size, offset=offset)
    binary_local = mito_im_blur > local_thresh
    binary_nobg = binary_local * (mito_im_blur >= mito_im_blur_thresh)
    binary_nosmall = remove_small_objects(binary_nobg, min_size=min_size)
    labeled_mito_mask = watershed(binary_nosmall, markers=labeled_mito_seeds, mask=binary_nosmall)

    return labeled_mito_mask

def remove_masks_on_edges(masks):
    """
    Given 2d image with labeled regions, remove those regions which contact any edges
    """
    new_masks = deepcopy(masks)
    for icell in np.unique(masks)[1:]:
        icell_mask = masks == icell
        sum_true_px_border = np.sum(icell_mask[0,:]) + np.sum(icell_mask[-1,:]) + np.sum(icell_mask[:,0]) + np.sum(icell_mask[:,-1])
        if sum_true_px_border > 0:
            new_masks[new_masks == icell] = 0
    return new_masks

def match_intensities_from_otsu(img1, img2):

    t1 = threshold_otsu(img1)
    t2 = threshold_otsu(img2)

    bright1 = img1[img1 > t1]
    bright2 = img2[img2 > t2]

    m1 = np.median(bright1)
    m2 = np.median(bright2)

    return img2 * m1/m2

def remove_overlapping_cells(labels, thickness, thickness_threshold=31, num_discard_px_threshold=50):
    """
    """

    pruned_labels = deepcopy(labels)
    for ilabel in np.unique(labels)[1:]:
        ilabel_mask = labels == ilabel
        thickness_values = thickness[ilabel_mask]
        num_discard_px = np.count_nonzero(thickness_values >= thickness_threshold)
        if num_discard_px >= num_discard_px_threshold:
            pruned_labels[ilabel_mask] = 0

    return pruned_labels

def cell_thickness_from_otsu(im, seg, sigma=5):
    is_cell = seg > 0

    im_blur_xy = np.array([gaussian(im_slice, sigma=[sigma,sigma], preserve_range=True) for im_slice in im])
    im_binary = im_blur_xy >= threshold_otsu(im_blur_xy)
    
    thickness = np.sum(im_binary, axis=0)
    
    return im_blur_xy, thickness, np.multiply(thickness, is_cell)

def cell_thickness(im, seg, sigma=5, bg_thresh=10):
    is_cell = seg > 0
    im_blur_xy = np.array([gaussian(im_slice, sigma=[sigma,sigma], preserve_range=True) for im_slice in im])
    thickness = nb_calc_thickness(im_blur_xy[:,:,:], bg_thresh=bg_thresh)
    return im_blur_xy, thickness, np.multiply(thickness, is_cell)

def FWHM(Y,X=None, background=None, bg_thresh=20, factor_decrease=2):

    if X is None:
        X = np.arange(len(Y))

    if background is None:
        background = np.min(Y)

    # logic from https://stackoverflow.com/questions/49480148/best-way-to-apply-a-function-to-a-slice-of-a-3d-numpy-array
    Y_bgsub = Y - background

    half_max = max(Y_bgsub) / factor_decrease
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - Y_bgsub[0:-1]) - np.sign(half_max - Y_bgsub[1:])

    #find the left and right most indexes
    a = X[:-1][d > 0]
    if len(a) == 0:
        return len(Y)
    else:
        left_idx = a[0]

    b = X[1:][d < 0]
    if len(b) == 0:
        return len(Y)
    else:
        right_idx = b[-1]

    if (left_idx == right_idx):
        return 0
    if np.mean(Y[left_idx:right_idx]) < bg_thresh:
        return 0

    return X[right_idx] - X[left_idx] #return the difference (full width)


@nb.njit
def nb_calc_thickness(im, bg_thresh=20):
    thick = np.zeros(im.shape[1:])
    for iy in np.arange(im.shape[1]):
        for ix in np.arange(im.shape[2]):
            Y = im[:,iy,ix]
            X = np.arange(len(Y))
            background = np.min(Y)

            # logic from https://stackoverflow.com/questions/49480148/best-way-to-apply-a-function-to-a-slice-of-a-3d-numpy-array
            Y_bgsub = Y - background

            half_max = max(Y_bgsub) / 2.
            #find when function crosses line half_max (when sign of diff flips)
            #take the 'derivative' of signum(half_max - Y[])
            d = np.sign(half_max - Y_bgsub[0:-1]) - np.sign(half_max - Y_bgsub[1:])

            #find the left and right most indexes
            a = X[:-1][d > 0]
            if len(a) == 0:
                thick[iy,ix] = len(Y)
                continue
            else:
                left_idx = a[0]

            b = X[1:][d < 0]
            if len(b) == 0:
                thick[iy,ix] = len(Y)
                continue
            else:
                right_idx = b[-1]

            Y_bgsub_within = Y_bgsub[left_idx:(right_idx+1)]
            if len(Y_bgsub_within) == 0:
                thick[iy,ix] = 0
            elif np.mean(Y_bgsub_within) < bg_thresh:
                thick[iy,ix] = 0
            else:
                thick[iy,ix] = X[right_idx] - X[left_idx] #return the difference (full width)
    return thick



def icellstack_mask_props(masks):

    nzslices = masks.shape[0]
    icells = np.unique(masks)[1:]
    ncells = len(icells)

    icell_2dmasks = np.zeros((ncells,) + masks.shape[1:])
    icell_z_profiles = np.zeros((ncells, nzslices))
    icell_nnz_zs = np.zeros((ncells,))

    for i in trange(ncells):

        icell = icells[i]

        icell_stack = (masks==icell)

        # all xy pixels in which icell shows up, regardless of z
        # icell_2dunion = np.sum(icell_stack, axis=0, dtype=bool)
        # icell_2dmasks.append(icell_2dunion)

        # number of icell pixels in each z-slice
        icell_z_profile = icell_stack.sum(axis=2).sum(axis=1)
        # print(icell_z_profile.shape, icell_z_profiles[icell,].shape)
        icell_z_profiles[i,] = icell_z_profile

        # number of z-slices in which icell shows up
        icell_nnz_z = np.count_nonzero(icell_z_profile)
        icell_nnz_zs[i,] = icell_nnz_z

        icell_2dmasks[i,] = icell_stack[np.argmax(icell_z_profile)]

    return icell_2dmasks, icell_z_profiles, icell_nnz_zs

def gaussian_blur_seg_mask(img_yx, sigma=3, preserve_range=False, threshold=0):
    blur = gaussian(img_yx, sigma=sigma, preserve_range=preserve_range)
    use_px = blur > threshold
    return use_px

def get_bounding_box(X, buffer=1):
    """ Calculates the bounding box of a ndarray"""
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
        # bbox.append(slice(idx_i[0]+1-buffer, idx_i[1]+1+buffer))
    return tuple(bbox)

def plot_contours(
        contour_image,
        show_image=None,
        contour_lo_value=40,
        contour_hi_value=100,
        fig=None, ax=None,
        figsize=(10,10),
        show_ticks=False,
        show=True, show_colorbar=True,
        cmap=plt.cm.gray,
        contour_lo_color="tab:green",
        contour_hi_color="crimson",
        contour_alpha=0.3,
    ):
    """
    Find contours at a constant value, display the image and plot all contours found. Returns handle to figure, handle to axis, and mask.
    """
    mask = np.logical_and(contour_image >= contour_lo_value, contour_image <= contour_hi_value)


    create_new_fig = (fig is None) and (ax is None)

    if create_new_fig:
        fig, ax = plt.subplots(figsize=figsize)

    if show_image is not None:
        p = ax.imshow(show_image, cmap=cmap)

        if show_colorbar:
            plt.colorbar(p)

    contours = find_contours(contour_image, contour_lo_value)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=contour_lo_color, alpha=contour_alpha)

    contours = find_contours(contour_image, contour_hi_value)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=contour_hi_color, alpha=contour_alpha)

    if create_new_fig:
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        if show:
            plt.show()

    return fig, ax, mask


def calculate_segmentation_masks(intensity_dataset, seg_function):
    """
    Wrapper function to apply a segmentation function operating on yx images to a multidimensional xarray.DataArray
    """
    return xr.apply_ufunc(
        seg_function,
        intensity_dataset,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        # dask="allowed",
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

def create_rfclassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, max_depth=max_depth, max_samples=max_samples)

def train_rfclassifier(clf, label_layer, features_dir="features"):

    # fit using labeled data
    # therefore, load features only for those images that have been labeled
    labeled_frame_idxs = np.unique(np.where(label_layer.data != 0)[0])
    labeled_frame_labels = np.concatenate(
        (
            label_layer.data[labeled_frame_idxs],
    #         prev_fandl["cell_labels"]
        ), axis=0
    )

    labeled_frame_features = np.concatenate(
        (
            np.array([ np.load(Path(features_dir).joinpath(str(idx)+".npy")) for idx in labeled_frame_idxs]),
    #         prev_fandl["cell_features"]
        ), axis=0
    )

    # clf = future.fit_segmenter(labeled_frame_labels, labeled_frame_features, clf)

    return future.fit_segmenter(labeled_frame_labels, labeled_frame_features, clf)

def features_func(sigma_min=1, sigma_max=16):
    return partial(multiscale_basic_features,
                intensity=True, edges=False, texture=True,
                sigma_min=sigma_min, sigma_max=sigma_max,
                # channel_axis=-1)
                # channel_axis=None)
                )

def calc_features_zstack(da_intensity, store_loc, overwrite=False, n_jobs=6):

    Path(store_loc).mkdir(exist_ok=True)

    def compute_and_save_features(i):

        features_npy_fname = Path(store_loc).joinpath(str(i) + ".npy")
        if (not features_npy_fname.exists()) or overwrite:

            im = da_intensity.isel(Z=i).compute().data
            im_channel_last_axis = np.moveaxis(im, 0, -1)

            np.save(
                features_npy_fname,
                features_func()(im_channel_last_axis)
            )

    Parallel(n_jobs=n_jobs)(delayed(compute_and_save_features)(i) for i in trange(len(da_intensity["Z"])))

# def calc_features(da_intensity, store_loc, overwrite=False, n_jobs=6, sep_img_key="file_info"):
#
#     da_intensity = da_intensity.astype(np.float32).chunk(chunks={
#         "file_info": 1,
#         # "channel": 1,
#         "y": len(da_intensity.y.data),
#         "x": len(da_intensity.x.data),
#     })
#     imgs = da_intensity[sep_img_key]
def calc_features(imgs, store_loc, overwrite=False, n_jobs=1):

    Path(store_loc).mkdir(exist_ok=True)

    def compute_and_save_features(i):
        features_npy_fname = Path(store_loc).joinpath(str(i) + ".npy")
        if (not features_npy_fname.exists()) or overwrite:
            np.save( features_npy_fname, features_func()(imgs[i]) )

    # for i in trange( len(imgs) ):
    #     compute_and_save_features(i)
    Parallel(n_jobs=n_jobs)(delayed(compute_and_save_features)(i) for i in trange( len(imgs) ))

def predict_prob_segmenter(features, clf):
    """Segmentation of images using a pretrained classifier.
    Parameters
    ----------
    features : ndarray
        Array of features, with the last dimension corresponding to the number
        of features, and the other dimensions are compatible with the shape of
        the image to segment, or a flattened image.
    clf : classifier object
        trained classifier object, exposing a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
        classifier must be already trained, for example with
        :func:`skimage.segmentation.fit_segmenter`.
    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier.
    """
    sh = features.shape

    if features.ndim > 2:
        features = features.reshape((-1, sh[-1]))

    try:
        predicted_labels = clf.predict_proba(features)
    except NotFittedError:
        raise NotFittedError(
            "You must train the classifier `clf` first"
            "for example with the `fit_segmenter` function."
        )
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(
                err.args[0] + '\n' +
                "Maybe you did not use the same type of features for training the classifier."
                )
    output = predicted_labels.reshape(sh[:-1] + (clf.n_classes_,))
    return output


def load_features(intensity_image_spec, df_filepaths):

    if isinstance(intensity_image_spec, Path):
        features_path = df_filepaths.loc[df_filepaths["intensity image"] == intensity_image_spec, "image features"].values[0]
    elif isinstance(intensity_image_spec, np.int32) or isinstance(intensity_image_spec, np.int64) or isinstance(intensity_image_spec, int):
        features_path = df_filepaths.loc[intensity_image_spec, "image features"]
    else:
        raise TypeError("Invalid type for intensity_image_spec")
        return

    return h5_to_dict(features_path)["features"]


def extract_useful_slices(df_filepaths, cell_labels_filepath, mito_labels_filepath, min_features_and_labels_filepath):


    if not min_features_and_labels_filepath.is_file():
        with h5py.File(min_features_and_labels_filepath, "w") as hf:

            all_labels = h5_to_dict(cell_labels_filepath)["cell_labels"]
            labeled_frame_idxs = np.unique(np.where(all_labels != 0)[0])
            cell_labeled_frame_labels = all_labels[labeled_frame_idxs]
            cell_labeled_frame_features = np.array([load_features(idx, df_filepaths) for idx in labeled_frame_idxs])

            hf.create_dataset("cell_labels", data=cell_labeled_frame_labels)
            hf.create_dataset("cell_features", data=cell_labeled_frame_features)

            all_labels = h5_to_dict(mito_labels_filepath)["mito_labels"]
            labeled_frame_idxs = np.unique(np.where(all_labels != 0)[0])
            mito_labeled_frame_labels = all_labels[labeled_frame_idxs]
            mito_labeled_frame_features = np.array([load_features(idx, df_filepaths) for idx in labeled_frame_idxs])

            hf.create_dataset("mito_labels", data=mito_labeled_frame_labels)
            hf.create_dataset("mito_features", data=mito_labeled_frame_features)
    else:
        raise RuntimeError("File already exists.")

@nb.jit(nopython=True)
def zmode(masks_):
    masks_zmode = np.zeros_like(masks_[0])

    for iy in range(masks_.shape[1]):
        for ix in range(masks_.shape[2]):

            curr_zprofile = masks_[:,iy,ix]
            curr_nz_zprofile = curr_zprofile[curr_zprofile > 0]

            if len(curr_nz_zprofile) > 0:
                # using return_counts doesn't seem to be compatible with numba
                # vals, cnts = np.unique(curr_nz_zprofile, return_counts=True)
                vals = np.unique(curr_nz_zprofile)
                cnts = np.array([np.sum(curr_nz_zprofile == val) for val in vals])
                masks_zmode[iy,ix] = vals[cnts.argmax()]
    return masks_zmode