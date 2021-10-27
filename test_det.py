""" Test the determinant function """

import numpy as np
import pytest
from det import det


@pytest.fixture(scope="class")
def two_by_two() -> np.ndarray:
    """
    2 x 2 matrix test fixture
    """
    return np.array([[2, 2], [3, 4]])

@pytest.fixture(scope="class")
def three_by_three() -> np.ndarray:
    """
    3 x 3 matrix test fixture
    """
    return np.random.uniform(0, 10, (3, 3))

@pytest.fixture(scope="class")
def four_by_four() -> np.ndarray:
    """
    4 x 4 matrix test fixture
    """
    return np.random.uniform(0, 10, (4, 4))

@pytest.fixture(scope="class")
def ten_by_ten() -> np.ndarray:
    """
    10 x 10 matrix test fixture
    """
    return np.random.uniform(0, 10, (10, 10))


@pytest.mark.usefixtures("two_by_two", "ten_by_ten")
class TestDet:
    """
    Tests for the determinant function
    """

    def test_two_by_two(self, two_by_two: np.ndarray):
        """
        Test to see if the value is correct for 2 x 2 matrix
        """

        got = det(two_by_two)
        want = np.linalg.det(two_by_two)
        assert np.isclose(got, want)

    def test_three_by_three(self, three_by_three: np.ndarray):
        """
        Test to see if value is correct for 3 x 3 matrix
        """

        got = det(three_by_three)
        want = np.linalg.det(three_by_three)

        assert np.isclose(got, want)

    def test_four_by_four(self, four_by_four: np.ndarray):
        """
        Test to see if value is correct for 4 x 4 matrix
        """

        got = det(four_by_four)
        want = np.linalg.det(four_by_four)

        assert np.isclose(got, want)

    def test_ten_by_ten(self, ten_by_ten: np.ndarray):
        """
        Test to see if value is correct for 10 x 10 matrix
        """

        got = det(ten_by_ten)
        want = np.linalg.det(ten_by_ten)

        assert np.isclose(got, want)
