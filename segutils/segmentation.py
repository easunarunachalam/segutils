from pathlib import Path
from skimage import feature

from skimage import segmentation, feature, future
from functools import partial
from joblib import Parallel, delayed

from .util import *

import numpy as np
import matplotlib.pyplot as plt

from skimage import filters, measure, segmentation, feature, future
from skimage.filters import gaussian
from sklearn.ensemble import RandomForestClassifier

from tqdm.notebook import tqdm



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

    for dx in tqdm(np.arange(-maxdxy, maxdxy+1)):
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

def gaussian_blur_seg_mask(img_yx, sigma=3, preserve_range=False, threshold=0):
    blur = gaussian(img_yx, sigma=sigma, preserve_range=preserve_range)
    use_px = blur > threshold
    return use_px

def get_bounding_box(x, buffer=1):
    """ Calculates the bounding box of a ndarray"""
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
        bbox.append(slice(idx_i[0]+1-buffer, idx_i[1]+1+buffer))
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


    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(figsize=figsize)

    if show_image is None:
        p = ax.imshow(contour_image, cmap=cmap)
    # elif show_image is False:
    #     continue
    else:
        p = ax.imshow(show_image, cmap=cmap)

    contours = measure.find_contours(contour_image, contour_lo_value)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=contour_lo_color, alpha=contour_alpha)

    contours = measure.find_contours(contour_image, contour_hi_value)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=contour_hi_color, alpha=contour_alpha)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_colorbar:
        plt.colorbar(p)

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
    return partial(feature.multiscale_basic_features,
                intensity=True, edges=False, texture=True,
                sigma_min=sigma_min, sigma_max=sigma_max,
                multichannel=True)

def calc_features(da_intensity, store_loc, overwrite=False, n_jobs=6):

    da_intensity = da_intensity.astype(np.float32).chunk(chunks={
        "file_info": 1,
        "channel": 1,
        "y": len(da_intensity.y.data),
        "x": len(da_intensity.x.data),
    })

    Path(store_loc).mkdir(exist_ok=True)

    np.save(Path(store_loc).joinpath("file_info"), da_intensity.file_info.data)

    def compute_and_save_features(i):

        features_npy_fname = Path(store_loc).joinpath(str(i) + ".npy")
        if (not features_npy_fname.exists()) or overwrite:

            im = da_intensity.isel(file_info=i).compute().data
            im_channel_last_axis = np.moveaxis(im, 0, -1)

            np.save(
                features_npy_fname,
                features_func()(im_channel_last_axis)
            )

    Parallel(n_jobs=n_jobs)(delayed(compute_and_save_features)(i) for i in trange(len(da_intensity["file_info"])))

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
