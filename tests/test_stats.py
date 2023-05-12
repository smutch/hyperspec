import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial import distance
from spectral.algorithms import spectral_angles

from hyperspec.stats import _cosine_similarity, pairwise_sam_similarity, pixelwise_sam_similarity


def test_cosine_similarity():
    # Define two example vectors
    vector1 = np.array([1, 2, 3], float)
    vector2 = np.array([4, 5, 6], float)

    # Calculate the expected cosine similarity using the scipy distance function
    expected_similarity = 1.0 - distance.cosine(vector1, vector2)

    # Call the function that calculates the cosine similarity
    actual_similarity = _cosine_similarity(vector1[np.newaxis, :], vector2[np.newaxis, :])[0]

    # Check that the actual similarity matches the expected similarity
    assert_array_almost_equal(actual_similarity, expected_similarity)


def test_pixelwise_sam_similarity():
    cube1 = np.random.rand(4, 4, 4)
    cube2 = np.ones(cube1.shape) * cube1[0, 0]
    sam1 = pixelwise_sam_similarity(cube1, cube2)
    sam2 = 1.0 - spectral_angles(cube1, cube2[0, 0][None, :]).squeeze()
    np.testing.assert_array_almost_equal(sam1, sam2)


def test_pairwise_sam_similarity():
    cube = np.random.rand(4, 4, 4)
    spectra = np.random.rand(4, 4)
    sam1 = pairwise_sam_similarity(cube, spectra)
    sam2 = 1.0 - spectral_angles(cube, spectra)
    np.testing.assert_array_almost_equal(sam1, sam2)
