import os
import sys
import numpy as np
from multiprocessing import Array

def matrix_sum_worker(start, size, matrix1, matrix2, result):
    for z in range(start, size, process_count):
        for y in range(size):
            for x in range(size):
                result[z*size*size + y*size + x] = matrix1[z][y][x] + matrix2[z][y][x]

if __name__ == '__main__':
    size = 2
    process_count = 1

    matrix1 = np.random.randint(1, 11, size=(size, size, size))
    matrix2 = np.random.randint(1, 11, size=(size, size, size))

    result = Array('i', size*size*size, lock=False)

    processes = []
    for process_num in range(process_count):
        pid = os.fork()
        if pid > 0:
            processes.append(pid)
        elif pid == 0:
            matrix_sum_worker(process_num, size, matrix1, matrix2, result)
            sys.exit(0)
        else:
            print(f"Fork failed with error code: {pid}")
            sys.exit(1)

    for p in processes:
        os.waitpid(p, 0)


    result_matrix = np.frombuffer(result, dtype=np.int32).reshape(size, size, size)
    
    print(matrix1)
    print("--------------------------")
    print(matrix2)
    print("--------------------------")
    print(result_matrix)

