import numpy as np
import numpy.typing as npt
import xarray as xr
from sklearn import decomposition as decomp
from spectral.algorithms import spectral_angles

__all__ = ["pca", "pixelwise_cosine_similarity"]


def pca(cube: xr.DataArray, n_components: int = 3) -> xr.Dataset:
    """
    Computes principal components of a cube.
    Args:
      cube (xr.DataArray): The cube to compute principal components of.
      n_components (int): The number of components to compute.
    Returns:
      xr.Dataset: A dataset containing the principal components.
    Examples:
      >>> cube = xr.DataArray(np.random.rand(3, 3, 3))
      >>> pca(cube, n_components=2)
      <xarray.Dataset>
      Dimensions:  (band: 2)
      Coordinates:
        * band     (band) int64 0 1
      Data variables:
          0        (band) float64 0.541 0.8
          1        (band) float64 0.8 0.541
    """
    model = decomp.PCA(n_components=n_components)
    bands = cube.band.values
    X = (  # noqa: N806  <- sklearn norm
        cube.dropna("x", how="all").dropna("y", how="all").values.reshape((-1, bands.size))
    )
    model.fit_transform(X)
    components = xr.Dataset(
        {str(ii): (("band",), component) for ii, component in enumerate(model.components_)},
        coords={"band": bands},
    )
    return components


def _cosine_similarity(arr1: npt.NDArray[np.float_], arr2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Calculate the vector-wise cosine similarity between two arrays of vectors.

    NOTE: This function calculates just the diagonal component of `scipy.distance.cdist(..., metric='cosine')`.

    Arguments:
        arr1: An array of vectors with shape (N, M) where N is the number of vectors and M is the dimensionality of each
              vector.
        arr2: An array of vectors with the same shape (N, M) as arr1.

    Returns:
        An array of shape (N,) containing the cosine similarity between each pair of vectors.
    """
    uv = np.average(arr1 * arr2, axis=1)
    uu = np.average(np.square(arr1), axis=1)
    vv = np.average(np.square(arr2), axis=1)
    dist = np.fmax(np.fmin(1.0 - uv / np.sqrt(uu * vv), 2.0), 0)
    return 1.0 - dist


def pixelwise_cosine_similarity(cube1: npt.NDArray[np.float_], cube2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the cosine similarity between two cubes.
    Args:
      cube1 (npt.NDArray[np.float_]): The first cube.
      cube2 (npt.NDArray[np.float_]): The second cube.
    Returns:
      npt.NDArray[np.float_]: An array of shape (N,) containing the cosine similarity between each pair of vectors.
    Raises:
      ValueError: If cube1 and cube2 have different shapes.
    Examples:
      >>> cube1 = np.random.rand(3, 3, 3)
      >>> cube2 = np.random.rand(3, 3, 3)
      >>> pixelwise_cosine_similarity(cube1, cube2)
      array([[0.988, 0.988, 0.988],
             [0.988, 0.988, 0.988],
             [0.988, 0.988, 0.988]])
    """
    arr1 = cube1.reshape((-1, cube1.shape[-1]))
    arr2 = cube2.reshape((-1, cube2.shape[-1]))

    if arr1.shape != arr2.shape:
        _err = f"cube1 and cube2 must have the same shape, but got {arr1.shape} and {arr2.shape}"
        raise ValueError(_err)

    return _cosine_similarity(arr1, arr2).reshape(cube1.shape[:-1])


def sam(arr1: npt.NDArray[np.float_], arr2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the cosine similarity between two 3D arrays.
    Args:
      arr1 (npt.NDArray[np.float_]): The first 3D array.
      arr2 (npt.NDArray[np.float_]): The second 3D array.
    Returns:
      npt.NDArray[np.float_]: The cosine similarity between arr1 and arr2.
    Notes:
      The result is clipped between -1.0 and 1.0.
    Examples:
      >>> sam(arr1, arr2)
      array([[0.9, 0.8],
             [0.7, 0.6]])
    """
    numerator = np.einsum("ijk,ijk->ij", arr1, arr2)
    denom1 = np.einsum("ijk,ijk->ij", arr1, arr1)
    np.sqrt(denom1, out=denom1)
    denom2 = np.einsum("ijk,ijk->ij", arr2, arr2)
    np.sqrt(denom2, out=denom2)
    result = numerator / (denom1 * denom2)
    np.clip(result, -1.0, 1.0, out=result)
    np.arccos(result, out=result)
    return result.squeeze()


def pairwise_sam(cube: npt.NDArray[np.float_], spectra: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the pairwise SAM between a cube and array of spectra.
    Args:
      cube (npt.NDArray[np.float_]): A 3-dimensional cube.
      spectra (npt.NDArray[np.float_]): A 1- or 2-dimensional array of spectra.
    Returns:
      npt.NDArray[np.float_]: The pairwise SAM between cube and spectra.
    Raises:
      ValueError: If cube does not have 3 dimensions or spectra does not have ≤2 dimensions.
    Examples:
      >>> cube = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      >>> spectra = np.array([[1, 2], [3, 4]])
      >>> pairwise_sam(cube, spectra)
      array([[0.        , 0.        ],
             [1.57079633, 1.57079633]])
    """
    if cube.ndim != 3:
        _err = f"cube must have 3 dimensions, but got {cube.ndim}"
        raise ValueError(_err)
    if spectra.ndim > 2:
        _err = f"spectra must have ≤2 dimensions, but got {spectra.ndim}"
        raise ValueError(_err)
    if spectra.ndim == 1:
        spectra = spectra[None, :]
    return spectral_angles(cube, spectra)


def pixelwise_sam(arr1: npt.NDArray[np.float_], arr2: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    Computes the pixelwise spherical angle measure (SAM) between two cubes / arrays with dim <= 3.
    Args:
      arr1 (npt.NDArray[np.float_]): The first cube / array.
      arr2 (npt.NDArray[np.float_]): The second cube / array.
    Returns:
      npt.NDArray[np.float_]: The SAM between the two cubes / arrays.
    Raises:
      ValueError: If arr1 and arr2 do not have the same shape.
    Examples:
      >>> arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      >>> arr2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
      >>> pixelwise_sam(arr1, arr2)
      array([[0., 0.],
             [0., 0.]])
    """
    if arr1.shape != arr2.shape or not arr1.ndim <= 3:
        _err = f"arr1 and arr2 must have the same shape and have ≤3 dimensions, but got {arr1.shape} and {arr2.shape}"
        raise ValueError(_err)
    if arr1.ndim == 1:
        arr1 = arr1[None, None, :]
        arr2 = arr2[None, None, :]
    if arr1.ndim == 2:
        arr1 = arr1[:, None, :]
        arr2 = arr2[:, None, :]
    return sam(arr1, arr2)
