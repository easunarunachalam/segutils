# segutils
generic cell/organelle segmentation utilities, mostly wrappers for skimage functions

# installation

```
conda create -n mitogenesis2
conda activate mitogenesis
pip install cellpose
pip install numpy scipy tqdm xarray matplotlib seaborn jupyterlab jupyterlab-widgets ipywidgets pandas napari[all] bfio aicsimageio scikit-image scikit-learn dask btrack
```


```
conda activate cellpose
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
python -m pip install cellpose
pip install xarray
pip install jupyterlab ipywidgets jupyterlab-widgets
pip install aicsimageio tqdm
pip install napari[all] btrack
pip install scikit-image scikit-learn matplotlib openpyxl
```