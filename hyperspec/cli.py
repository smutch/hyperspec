import json
from pathlib import Path
from typing import Union

import cv2
import holoviews as hv
import numpy as np
import typer
import xarray as xr

from hyperspec import registration
from hyperspec.io import read_cube, read_preview

app = typer.Typer()


@app.command()
def register(
    dst_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    src_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    crops_db: Path = typer.Argument(..., exists=True),  # noqa: B008
    out_path: Path = typer.Argument(...),  # noqa: B008
    *,
    smooth: float = 0.0,
    debug: bool = False,
):
    """
    Runs the registration process.

    Example:
        hyperspec register 2023-03-09_014/results/REFLECTANCE_2023-03-09_014.hdr 2023-03-09_015/results/REFLECTANCE_2023-03-09_015.hdr bounds.json registered.zarr
    """
    capture_id = src_path.parent.parts[-2]

    with open(crops_db) as f:
        crops = json.load(f)

    if capture_id not in crops:
        _err = f"Capture ID {capture_id} not found in crops file"
        raise ValueError(_err)

    crop_bounds = np.around(np.array(crops[capture_id][:4])).astype(int)

    src_preview = read_preview(src_path, bounds=crop_bounds, smooth=smooth, greyscale=True)
    dst_preview = read_preview(dst_path, bounds=crop_bounds, smooth=smooth, greyscale=True)
    src_cube = read_cube(src_path, bounds=crop_bounds, smooth=smooth)
    dst_cube = read_cube(dst_path, bounds=crop_bounds, smooth=smooth)

    result, result_preview, matched_vis = registration.register(dst_preview, dst_cube, src_preview, src_cube)
    if result is None or result_preview is None:
        _err = "Registration failed"
        raise ValueError(_err)

    if debug:
        cv2.imshow("Matched Keypoints", matched_vis)
        cv2.waitKey(0)

    cv2.imwrite(f"{out_path.parent}/{out_path.stem}-preview.png", result_preview)
    xr.Dataset({capture_id: result}).to_zarr(out_path.with_suffix(".zarr"), mode="w")


@app.command()
def crop(
    capture_dir: Path = typer.Argument(..., file_okay=False, exists=True),  # noqa: B008
    crop_db: Path = typer.Argument(..., dir_okay=False),  # noqa: B008
    *,
    capture_ids: Union[str, None] = None,
):
    """
    Start an interactive web interface for cropping cubes.

    Example:
        hyperspec crop set1 bounds.json
    """
    hv.extension("bokeh")  # type: ignore
    _capture_ids = capture_ids
    if _capture_ids is not None:
        _capture_ids = _capture_ids.split(",")

    registration.crop(capture_dir, _capture_ids, crop_db).show()
