import os
import sys
import numpy as np
from multiprocessing import Pipe

def matrix_multiply_worker(start, end, size, matrix1, matrix2, conn):
    result = np.zeros((size, size, size), dtype=int)
    for z in range(start, end):
        result[z] = np.matmul(matrix1[z], matrix2[z])
    conn.send(result)
    conn.close()

if __name__ == '__main__':
    size = 500
    process_count = 1

    matrix1 = np.random.randint(1, 11, size=(size, size, size))
    matrix2 = np.random.randint(1, 11, size=(size, size, size))
    result = np.zeros((size, size, size), dtype=int)

    processes = []
    connections = []
    step = size // process_count

    for process_num in range(process_count):
        parent_conn, child_conn = Pipe()
        connections.append(parent_conn)
        start = process_num * step
        end = (process_num + 1) * step if process_num != process_count - 1 else size

        pid = os.fork()
        if pid > 0:
            processes.append(pid)
        elif pid == 0:
            matrix_multiply_worker(start, end, size, matrix1, matrix2, child_conn)
            sys.exit(0)
        else:
            print(f"Fork failed with error code: {pid}")
            sys.exit(1)

    for conn in connections:
        result += conn.recv()

    for p in processes:
        os.waitpid(p, 0)

"""
    print(matrix1)
    print("--------------------------")
    print(matrix2)
    print("--------------------------")
    print(result)
"""

