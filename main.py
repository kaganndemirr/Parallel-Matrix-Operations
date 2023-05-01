import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

N = 3

# CUDA kernel
kernel_code = """
__global__ void matrix_add(const int *a, const int *b, int *c, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = x * N * N + y * N + z;

    if (x < N && y < N && z < N) {
        c[index] = a[index] + b[index];
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)

# Host arrays
a = np.random.randint(1, 11, (N, N, N)).astype(np.int32)
b = np.random.randint(1, 11, (N, N, N)).astype(np.int32)
c = np.empty_like(a)

# Flatten the 3D arrays to 1D
a_flat = a.flatten()
b_flat = b.flatten()

# Device arrays
a_gpu = gpuarray.to_gpu(a_flat)
b_gpu = gpuarray.to_gpu(b_flat)
c_gpu = gpuarray.empty_like(a_gpu)

# Call the kernel
grid_dim = (1, 1, 1)
block_dim = (N, N, N)
matrix_add = mod.get_function("matrix_add")
matrix_add(a_gpu, b_gpu, c_gpu, np.int32(N), grid=grid_dim, block=block_dim)

# Copy the result back to host
c_flat = c_gpu.get()
c_result = np.reshape(c_flat, (N, N, N))

print("A:\n", a)
print("------------------------------")
print("B:\n", b)
print("------------------------------")
print("C:\n", c_result)
