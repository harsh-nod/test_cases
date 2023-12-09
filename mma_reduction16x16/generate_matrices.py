import numpy as np

M = 16
N = 16
K = 16

np.random.seed(0)

a = np.random.random_integers(0, 10, size=(M, K)) / (2 * 10.0)
b = np.random.random_integers(0, 10, size=(K, N)) / (2 * 10.0)
d = np.random.random_integers(0, 10, size=(M, N)) / (2 * 10.0)
a = np.float16(a)
b = np.float16(b)
d = a @ b.T
c = np.tile(np.maximum.reduce(d, axis=1), (N)).reshape(M, N).T
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
