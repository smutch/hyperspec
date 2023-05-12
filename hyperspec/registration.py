import json
import logging
from os import PathLike
from pathlib import Path
from typing import Any
from warnings import warn

import cv2
import holoviews as hv
import imutils
import numpy as np
import numpy.typing as npt
import panel as pn
import param
import xarray as xr

from hyperspec.io import read_preview

__all__ = ["register", "crop", "read_crop_db"]
logger = logging.getLogger(__name__)


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
    # NOTE: Order of points is important here to provide a valid contour
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transformed = cv2.perspectiveTransform(points[None, :, :], homog).squeeze()
    area = cv2.contourArea(transformed)
    tol = 0.1
    if area < 1.0 - tol or area > 1.0 + tol:
        _err = f"Homography does preserve area to within a {int(tol*100)}% ({area})"
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


class Cropper(param.Parameterized):
    image_selection = param.Selector()
    store_button = param.Action(lambda cropper: cropper.store_bounds(), label="Store bounds")

    def __init__(self, capture_dir: PathLike, capture_ids: list[str] | None, crop_db: PathLike | None):
        self.poly = hv.Polygons([]).opts(fill_alpha=0.2)
        self.poly_stream = hv.streams.PolyDraw(  # type: ignore
            source=self.poly,
            drag=False,
            num_objects=1,
            show_vertices=True,
            styles={"fill_color": "red", "line_color": "red"},
            vertex_style={"fill_color": "red", "fill_alpha": 0.5},
        )
        self.poly_edit_stream = hv.streams.PolyEdit(  # type: ignore
            source=self.poly, vertex_style={"fill_color": "red", "fill_alpha": 0.2}, shared=True
        )

        self.capture_dir = Path(capture_dir)

        if capture_ids is not None:
            self.cube_paths = {
                k: self.capture_dir.glob(f"**/results/REFLECTANCE_{k}.hdr").__next__() for k in capture_ids
            }
        else:
            paths = self.capture_dir.glob("**/results/REFLECTANCE_*.hdr")
            self.cube_paths = {p.stem.removeprefix("REFLECTANCE_"): p for p in paths}

        self.capture_ids = sorted(self.cube_paths.keys())

        self.param.image_selection.objects = self.capture_ids
        self.param.image_selection.default = self.capture_ids[0]

        self.crop_db = crop_db
        self.crop_corners = {}

        if crop_db is not None:
            try:
                self.crop_corners = read_crop_db(crop_db)
            except FileNotFoundError:
                logger.info("Crop database not found. Creating a new database.")

        super().__init__()

    def store_bounds(self):
        corners = (
            np.stack([self.poly_stream.data["xs"], self.poly_stream.data["ys"]], -1).astype(int)[:4].squeeze().tolist()
        )
        self.crop_corners[self.image_selection] = corners
        if self.crop_db is not None:
            with open(self.crop_db, "w") as fp:
                json.dump(self.crop_corners, fp)

    @param.depends("image_selection")
    def plot(self):
        verts = self.crop_corners.get(self.image_selection, [])
        if len(verts) > 0:
            x = np.array([v[0] for v in verts])
            y = np.array([v[1] for v in verts])
        else:
            x, y = [], []
        poly = {"x": x, "y": y}
        self.poly.data = [poly]  # type: ignore
        old_poly = hv.Polygons([poly]).opts(fill_color=None, line_color="orange", line_alpha=0.5)
        im = read_preview(self.cube_paths[self.image_selection], greyscale=False)  # type: ignore
        fig = (
            hv.RGB(np.flip(np.array(im), (0, -1)), bounds=(0, 0, *im.shape[:2]))
            .opts(invert_yaxis=True)
            .opts(height=700, width=700, aspect="equal")  # type: ignore
        )
        return fig * old_poly * self.poly  # type: ignore


def crop(capture_dir: PathLike, capture_ids: list[str] | None, crop_db: PathLike | None) -> pn.layout.Row:
    cropper = Cropper(capture_dir, capture_ids, crop_db)
    return pn.Row(cropper.param, cropper.plot)


def read_crop_db(crop_db: PathLike) -> dict[str, list[int]]:
    with open(crop_db) as fp:
        crop_corners = json.load(fp)
    return crop_corners
