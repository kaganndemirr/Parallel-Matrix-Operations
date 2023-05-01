import os
import sys
import numpy as np
from multiprocessing import Value, Array

def create_random_matrix(size):
    return np.random.randint(1, 11, (size, size, size))

def matrix_sum_worker(start, end, matrix1, matrix2, result, size):
    for idx in range(start, end):
        i = idx // (size * size)
        j = (idx % (size * size)) // size
        k = idx % size
        result[idx] = matrix1[i][j][k] + matrix2[i][j][k]

def main():
    size = 500

    matrix1 = create_random_matrix(size)
    matrix2 = create_random_matrix(size)

    process_count = 8
    total_elements = size * size * size
    elements_per_process = (total_elements + process_count - 1) // process_count

    result = Array('i', total_elements)

    processes = []
    for p in range(process_count):
        start = p * elements_per_process
        end = min((p + 1) * elements_per_process, total_elements)

        pid = os.fork()
        if pid == 0:
            matrix_sum_worker(start, end, matrix1, matrix2, result, size)
            sys.exit(0)
        elif pid > 0:
            processes.append(pid)
        else:
            print("Fork failed")
            sys.exit(1)

    for pid in processes:
        os.waitpid(pid, 0)

    result_matrix = np.array(result[:]).reshape(size, size, size)
    matrix1 = np.array(matrix1[:]).reshape(size, size, size)
    matrix2 = np.array(matrix2[:]).reshape(size, size, size)
    """
    print(matrix1)
    print("--------------------------")
    print(matrix2)
    print("--------------------------")
    print(result_matrix)
    """

if __name__ == "__main__":
    main()
