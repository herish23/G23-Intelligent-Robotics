import numpy as np
import yaml
from PIL import Image
from dataclasses import dataclass

@dataclass
class MapInfo:
    occupancy_grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    width: int
    height: int
    occupied_thresh: float
    free_thresh: float

## load map from pgm and yaml files
def load_map(pgm_path, yaml_path):
    # load yaml metadata
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    res = meta['resolution']
    origin = meta['origin']
    ox, oy = origin[0], origin[1]
    occ_thresh = meta.get('occupied_thresh', 0.65)
    free_thresh = meta.get('free_thresh', 0.25)
    negate = meta.get('negate', 0)

    # load pgm image
    img = Image.open(pgm_path)
    img_arr = np.array(img)

    # convert to occupancy grid - trinary mode
    # pgm: 255=white=free, 0=black=occupied
    norm = img_arr / 255.0
    occ_grid = np.where(norm > (1 - free_thresh), 0.0,
                        np.where(norm < occ_thresh, 1.0, 0.5))

    h, w = occ_grid.shape

    return MapInfo(
        occupancy_grid=occ_grid,
        resolution=res,
        origin_x=ox,
        origin_y=oy,
        width=w,
        height=h,
        occupied_thresh=occ_thresh,
        free_thresh=free_thresh
    )
