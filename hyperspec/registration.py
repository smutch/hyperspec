import json
import logging
from pathlib import Path
from typing import Any
from warnings import warn

import cv2
import imutils
import numpy as np
import numpy.typing as npt
import typer
import xarray as xr

from hyperspec.io import read_cube, read_preview

__all__ = ["register"]
logger = logging.getLogger(__name__)


def _cli(
    dst_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    src_path: Path = typer.Argument(..., dir_okay=False, exists=True),  # noqa: B008
    crops_path: Path = typer.Argument(..., exists=True),  # noqa: B008
    out_path: Path = typer.Argument(...),  # noqa: B008
    *,
    smooth: float = 0.0,
    debug: bool = False,
):
    """
    Runs the registration process.
    Args:
      dst_path (Path): Path to the destination cube.
      src_path (Path): Path to the source cube.
      crops_path (Path): Path to the crops file.
      out_path (Path): Path to the output file.
      smooth (float): Smoothing factor for the previews.
      debug (bool): Flag to show the matched keypoints.
    Returns:
      None
    Side Effects:
      Writes the output file to the given path.
    """
    capture_id = src_path.parent.parts[-2]

    with open(crops_path) as f:
        crops = json.load(f)

    if capture_id not in crops:
        _err = f"Capture ID {capture_id} not found in crops file"
        raise ValueError(_err)

    crop_bounds = np.around(np.array(crops[capture_id][:4])).astype(int)

    src_preview = read_preview(src_path, bounds=crop_bounds, smooth=smooth, greyscale=True)
    dst_preview = read_preview(dst_path, bounds=crop_bounds, smooth=smooth, greyscale=True)
    src_cube = read_cube(src_path, bounds=crop_bounds, smooth=smooth)
    dst_cube = read_cube(dst_path, bounds=crop_bounds, smooth=smooth)

    result, result_preview, matched_vis = register(dst_preview, dst_cube, src_preview, src_cube)
    if result is None or result_preview is None:
        _err = "Registration failed"
        raise ValueError(_err)

    if debug:
        cv2.imshow("Matched Keypoints", matched_vis)
        cv2.waitKey(0)

    cv2.imwrite(f"{out_path.parent}/{out_path.stem}-preview.png", result_preview)
    xr.Dataset({capture_id: result}).to_zarr(out_path.with_suffix(".zarr"), mode="w")


def validate_homography(homog: npt.NDArray[np.float_]):
    logger.info(f"Validating homography for {homog}")
    # must preserves orientation
    if np.linalg.det(homog[:2, :2]) < 0:
        _err = "Homography does not preserve orientation"
        raise ValueError(_err)

    # check the determinant is non-zero
    if np.isclose(np.linalg.det(homog), 0.0):
        _err = "Homography transform is non-invertable"
        raise ValueError(_err)

    # check the transform is homogeneous
    if homog[2, 2] != 1.0:
        _err = "Homography transform is not homogeneous"
        raise ValueError(_err)

    # must approximately preserve area (to within a tolerance appropriate for us)
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    transformed = cv2.perspectiveTransform(points[None, :, :], homog).squeeze()
    area = cv2.contourArea(transformed)
    tol = 0.1
    if area < 1.0 - tol or area > 1.0 + tol:
        _err = f"Homography does preserve area to within a {int(tol*100)}%"
        raise ValueError(_err)

    # we do not want large perspective shifts (assuming images are taken almost front on)
    if np.any(np.abs(homog[2, :2]) > 0.001):
        _err = "Homography results in large perspective shift"
        raise ValueError(_err)


def register(
    dst_preview: npt.NDArray,
    dst_cube: xr.DataArray,
    src_preview: npt.NDArray,
    src_cube: xr.DataArray,
    *,
    orb_create_kwargs: dict[str, Any] | None = None,
    flann_index_kwargs: dict[str, Any] | None = None,
    flann_search_kwargs: dict[str, Any] | None = None,
) -> tuple[xr.DataArray | None, npt.NDArray | None, npt.NDArray]:
    """
    Registers the source cube to the destination cube.
    Args:
      dst_preview (npt.NDArray): Preview of the destination cube.
      dst_cube (xr.DataArray): Destination cube.
      src_preview (npt.NDArray): Preview of the source cube.
      src_cube (xr.DataArray): Source cube.
      orb_create_kwargs (dict[str, Any] | None): Keyword arguments for ORB creation.
      flann_index_kwargs (dict[str, Any] | None): Keyword arguments for FLANN index.
      flann_search_kwargs (dict[str, Any] | None): Keyword arguments for FLANN search.
    Returns:
      tuple[xr.DataArray | None, npt.NDArray | None, npt.NDArray]: The registered cube, the registered preview, and the
                                                                   matched keypoints visualization.
    Side Effects:
      None
    Examples:
      >>> register(dst_preview, dst_cube, src_preview, src_cube,
                   orb_create_kwargs={"nfeatures": 1000},
                   flann_index_kwargs={"algorithm": 5})
      (xr.DataArray, npt.NDArray, npt.NDArray)
    """
    _orb_create_kwargs = {"nfeatures": 10_000, "scaleFactor": 1.2, "scoreType": cv2.ORB_HARRIS_SCORE}
    _orb_create_kwargs.update(orb_create_kwargs or {})
    orb = cv2.ORB_create(**_orb_create_kwargs)

    keypoints_src, descriptors_src = orb.detectAndCompute(src_preview, None)
    keypoints_dst, descriptors_dst = orb.detectAndCompute(dst_preview, None)

    _flann_index_kwargs = {"algorithm": 6, "table_number": 6, "key_size": 10, "multi_probe_level": 2}
    _flann_index_kwargs.update(flann_index_kwargs or {})
    _flann_search_kwargs = {"checks": 50}
    _flann_search_kwargs.update(flann_search_kwargs or {})
    matcher = cv2.FlannBasedMatcher(_flann_index_kwargs, _flann_search_kwargs)

    matches = [m for m, n in matcher.knnMatch(descriptors_src, descriptors_dst, k=2) if m.distance < 0.7 * n.distance]

    matched_vis = cv2.drawMatches(src_preview, keypoints_src, dst_preview, keypoints_dst, matches, None)
    matched_vis = imutils.resize(matched_vis, width=1_000)

    try:
        pts_src = np.array([keypoints_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_dst = np.array([keypoints_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homog, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        validate_homography(homog)

        result_preview = cv2.warpPerspective(src_preview, homog, src_preview.shape[:2][::-1])

        result = xr.zeros_like(dst_cube)
        for band in result.band:
            result.loc[..., band] = cv2.warpPerspective(
                src_cube.sel(band=band).values, homog, dst_preview.shape[:2][::-1], borderValue=-999
            )
        result = xr.DataArray(result, dims=dst_cube.dims, coords=dst_cube.coords)
    except cv2.error as err:
        warn(err.msg, stacklevel=2)
        return None, None, matched_vis

    return result, result_preview, matched_vis


if __name__ == "__main__":
    typer.run(_cli)
