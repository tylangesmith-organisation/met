""" Test the determinant function """

import numpy as np
import pytest
from det import det


@pytest.fixture(scope="class")
def two_by_two() -> np.ndarray:
    """
    Two by two matrix test fixture
    """
    return np.array([[2, 2], [3, 4]])


@pytest.fixture(scope="class")
def ten_by_ten() -> np.ndarray:
    """
    Ten by 10 matrix test fixture
    """
    return np.random.uniform(0, 10, (10, 10))


@pytest.mark.usefixtures("two_by_two", "ten_by_ten")
class TestDet:
    """
    Tests for the determinant function
    """

    def test_two_by_two(self, two_by_two: np.ndarray):
        """
        Test to see if the value is correct for two by two matrix
        """

        got = det(two_by_two)
        want = np.linalg.det(two_by_two)
        assert np.isclose(got, want)

    def test_ten_by_ten(self, ten_by_ten: np.ndarray):
        """
        Test to see if value is correct for ten by ten matrix
        """

        got = det(ten_by_ten)
        want = np.linalg.det(two_by_two)

        assert np.isclose(got, want)
