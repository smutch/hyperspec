# ruff: noqa: T201
import numpy as np

import hyperspec as hs

original_path = "../../data/set1/2023-03-09_014/results/REFLECTANCE_2023-03-09_014.hdr"
original_bounds = np.array(
    [
        [49.70, 157.82],
        [472.23, 153.13],
        [471.23, 471.60],
        [50.70, 472.54],
    ],
    dtype=int,
)
capture_path = "../../data/set1/2023-03-09_015/results/REFLECTANCE_2023-03-09_015.hdr"
capture_bounds = np.array(
    [
        [50.70, 157.82],
        [473.22, 153.13],
        [471.23, 472.54],
        [49.70, 474.42],
    ],
    dtype=int,
)

original_cube = hs.io.read_cube(original_path, bounds=original_bounds)
original_preview = hs.io.read_preview(original_path, bounds=original_bounds)
capture_cube = hs.io.read_cube(capture_path, bounds=capture_bounds)
capture_preview = hs.io.read_preview(capture_path, bounds=capture_bounds)

result, result_preview, matched_vis = hs.registration.register(
    capture_preview, capture_cube, original_preview, original_cube
)

print("Successfully registered!")
