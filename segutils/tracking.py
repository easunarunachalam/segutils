import btrack
import numpy as np
import pandas as pd
from skimage.util import map_array

def btrack_track_masks(da_seg, config_file):

    objects = btrack.utils.segmentation_to_objects(da_seg.data, properties=('area',))

    # initialise a tracker session using a context manager
    with btrack.BayesianTracker() as tracker:

      # configure the tracker using a config file
      tracker.configure_from_file(config_file)

      # append the objects to be tracked
      tracker.append(objects)

      # set the volume (Z axis volume is set very large for 2D data)
      tracker.volume=((0, da_seg.X.shape[0]), (0, da_seg.Y.shape[0]), (-1e5, 1e5))

      # track them (in interactive mode)
      tracker.track_interactive(step_size=100)

      # generate hypotheses and run the global optimizer
      tracker.optimize()

      # store the data in an HDF5 file
      # tracker.export("tracks.h5", obj_type="obj_type_1")

      # get the tracks as a python list
      tracks = tracker.tracks

      # optional: get the data in a format for napari
      data, properties, graph = tracker.to_napari(ndim=2)

    # pd.DataFrame([obj.to_dict() for obj in objects])

    # viewer.add_tracks(data, properties=properties, graph=graph)

    df_tracks = pd.concat([pd.DataFrame(track.to_dict()) for track in tracks], ignore_index=True)

    def track_timeseries_labels(segmentation, tracks):
        track_positions = tracks.loc[~tracks.dummy,['ID','t','y','x']]
        relabeled = np.zeros_like(segmentation)
        for t, df in track_positions.groupby('t'):
            single_segmentation = segmentation[t]
            new_id, tc, yc, xc = tuple(np.round(df.values).astype(int).T)
            old_id = single_segmentation[yc,xc]
            relabeled[t] = (single_segmentation>0)*map_array(single_segmentation, old_id, new_id)
        return relabeled

    relabeled_masks = track_timeseries_labels(da_seg.data.astype(np.uint16), df_tracks)

    return relabeled_masks
