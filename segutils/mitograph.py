import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import imageio as iio

import matplotlib.pyplot as plt

from .segmentation import get_bounding_box

def export_stacks_for_mitograph_single_timepoint(labels_2d_img, da_cyto, da_mito, min_size_threshold=50, min_dim_threshold=10, bg_intensity=99, scene=0, dry_run=True):
    
    n_images = 0

    cell_labels = np.unique(labels_2d_img[labels_2d_img > 0])
    for ilabel in tqdm(cell_labels, desc="export stacks", leave=False):
    
        is_curr_icell = labels_2d_img == ilabel

        # exclude overly small cells
        if np.sum(is_curr_icell) < min_size_threshold:
            continue
        
        # plt.imshow(is_curr_icell)
        # plt.show()

        try:
            y_bb, x_bb = get_bounding_box(is_curr_icell)
            curr_bb_mask = is_curr_icell[y_bb,x_bb]
            
            if not dry_run:
                
                # save network image stacks
                curr_bb_mitostack = da_mito.data[:, y_bb, x_bb]
                curr_bb_masked_mitostack = np.multiply(
                    np.tile(curr_bb_mask, (curr_bb_mitostack.shape[0],1,1)),
                    curr_bb_mitostack
                )
                background = np.random.randn(*(curr_bb_masked_mitostack.shape))*2 + bg_intensity
                curr_bb_masked_mitostack[curr_bb_masked_mitostack == 0] = background[curr_bb_masked_mitostack == 0]
        
                if (curr_bb_masked_mitostack.shape[1] < min_dim_threshold) or (curr_bb_masked_mitostack.shape[2] < min_dim_threshold):
                    continue
        
                tif_path = f"extracted_networks/S={scene:d}/S={scene:d}_cell={ilabel:d}.tif"
                Path(tif_path).parent.mkdir(parents=True, exist_ok=True)
                iio.volwrite(tif_path, curr_bb_masked_mitostack)
        
        
                # save masked cell stacks
                curr_bb_cytostack = da_cyto.data[:, y_bb, x_bb]
                curr_bb_masked_cytostack = np.multiply(
                    np.tile(curr_bb_mask, (curr_bb_cytostack.shape[0],1,1)),
                    curr_bb_cytostack
                )
        
                tif_path = f"extracted_cells/S={scene:d}/S={scene:d}_cell={ilabel:d}_maskpxonly.tif"
                Path(tif_path).parent.mkdir(parents=True, exist_ok=True)
                iio.volwrite(tif_path, curr_bb_masked_cytostack)

                
                # save cell image stacks
                tif_path = f"extracted_cells/S={scene:d}/S={scene:d}_cell={ilabel:d}.tif"
                Path(tif_path).parent.mkdir(parents=True, exist_ok=True)
                iio.volwrite(tif_path, curr_bb_cytostack)
            
            n_images += 1
        except:
            continue
        
    return n_images