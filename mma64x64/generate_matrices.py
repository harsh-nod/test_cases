import numpy as np

M = 16 * 4
N = 16 * 4
K = 16 * 4

np.random.seed(0)

a = np.random.random_integers(0, 10, size=(M, K)) / (2 * 10.0)
b = np.random.random_integers(0, 10, size=(K, N)) / (2 * 10.0)
a = np.float16(a)
b = np.float16(b)
c = a @ b.T
print(a)
print(b)
print(c)

# Write matrices
with open(f'matrix_a.npy', 'wb') as f:
    np.save(f, a)
with open(f'matrix_b.npy', 'wb') as f:
    np.save(f, b)
with open(f'matrix_c.npy', 'wb') as f:
    np.save(f, c)
