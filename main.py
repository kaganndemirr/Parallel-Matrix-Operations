import numpy as np
from numba import cuda

size = 500


@cuda.jit
def matrix_add(a, b, c, size):
    x, y, z = cuda.grid(3)

    if x < size and y < size and z < size:
        index = x * size * size + y * size + z
        c[index] = a[index] + b[index]


# Host arrays
a = np.random.randint(1, 11, (size, size, size)).astype(np.int32)
b = np.random.randint(1, 11, (size, size, size)).astype(np.int32)
c = np.empty_like(a)

# Flatten the 3D arrays to 1D
a_flat = a.flatten()
b_flat = b.flatten()
c_flat = np.empty_like(a_flat)

# Device arrays
a_device = cuda.to_device(a_flat)
b_device = cuda.to_device(b_flat)
c_device = cuda.device_array_like(c_flat)

# Call the kernel
threadsperblock = (8, 8, 8)
blockspergrid_x = int(np.ceil(a.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(a.shape[1] / threadsperblock[1]))
blockspergrid_z = int(np.ceil(a.shape[2] / threadsperblock[2]))
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

matrix_add[blockspergrid, threadsperblock](a_device, b_device, c_device, np.int32(size))

# Copy the result back to host
c_flat = c_device.copy_to_host()
c_result = np.reshape(c_flat, (size, size, size))

# print("A:\n", a)
# print("B:\n", b)
# print("C:\n", c_result)
