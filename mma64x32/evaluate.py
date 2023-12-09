import numpy as np

with open('output.npy', 'rb') as f:
  output = np.load(f)

with open('matrix_c.npy', 'rb') as f:
  expected_output = np.load(f)

error = np.max(np.abs(output - expected_output))
print("Max error = ", error)
