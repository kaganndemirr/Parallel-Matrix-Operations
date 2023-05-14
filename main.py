import numpy as np
from numba import cuda


@cuda.jit
def gpu_matrix_multiply(size, matrix1, matrix2, result):
    x, y, z = cuda.grid(3)
    if x < size and y < size and z < size:
        for k in range(size):
            result[z, y, x] += matrix1[z, y, k] * matrix2[z, k, x]


def main():
    size = 2

    matrix1 = np.random.randint(1, 11, size=(size, size, size)).astype(np.int32)
    matrix2 = np.random.randint(1, 11, size=(size, size, size)).astype(np.int32)

    result = np.zeros((size, size, size), dtype=np.int32)

    threadsperblock = (8, 8, 8)
    blockspergrid_x = int(np.ceil(size / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(size / threadsperblock[1]))
    blockspergrid_z = int(np.ceil(size / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    gpu_matrix_multiply[blockspergrid, threadsperblock](size, matrix1, matrix2, result)

    print("A:\n", matrix1)
    print("B:\n", matrix2)
    print("C:\n", result)


if __name__ == '__main__':
    main()
