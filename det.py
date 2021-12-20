""" Function to calculate determinant of a matrix """
import numpy as np
from math import pow


def _calculate_two_by_two_det(m: np.ndarray) -> float:
    """
    Calculates the determinant of a 2 by 2 matrix
    [[a, b],
    [c, d]]
    (a*d) - (b*c)
    """
    return (m[0][0]*m[1][1]) - (m[1][0]*m[0][1])


def _get_submatrix(index: int, m: np.ndarray) -> np.ndarray:
    columns_to_keep = [column_index for column_index in range(len(m)) if column_index != index]
    return m[1:, columns_to_keep]


def _get_operation(index: int) -> int:
    return pow(-1, index)


def det(m: np.ndarray) -> float:
    """
    Calculates the determinant of a matrix.

    :param: m: numpy.ndarray guaranteed to be a valid matrix shape.

    :return: Determinant of the matrix
    """
    # Base cases
    if m.shape == (1, 1):
        return m[0]
    if  m.shape == (2, 2):
        return _calculate_two_by_two_det(m=m)

    # For dimensions > 2
    # Each element in the top row * it's submatrix determinant.
    # Summation of each result alternating (+, -)
    determinant = 0
    for index, element in enumerate(m[0]):
        submatrix = _get_submatrix(index=index, m=m)
        operation = _get_operation(index=index)
        determinant += operation * (element * det(submatrix))

    return determinant
