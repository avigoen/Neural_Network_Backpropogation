import numpy as np


def dot_product_batches(matrix1, matrix2, batch_size=500):
    # Calculate the number of batches
    num_batches = matrix1.shape[1] // batch_size

    if num_batches == 0:
        return _dot_product_normal(matrix1, matrix2)

    return _dot_product_batches(matrix1, matrix2, batch_size, num_batches)


def _dot_product_normal(matrix1, matrix2):
    if matrix1.shape[1] != matrix2.shape[0]:
        return np.dot(matrix1, matrix2.T)
    return np.dot(matrix1, matrix2)


def _dot_product_batches(matrix1, matrix2, batch_size, num_batches):
    # Initialize the result matrix
    result = np.zeros((matrix1.shape[0], matrix2.shape[1]))

    results = [np.dot(matrix1[:, (batch_idx * batch_size):(min((batch_idx + 1) * batch_size, matrix1.shape[1]))],
                      matrix2[(batch_idx * batch_size):(min((batch_idx + 1) * batch_size, matrix1.shape[1])), :]) for batch_idx in range(num_batches)]
    result = np.sum(results, axis=0)
    return result
