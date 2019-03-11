import numpy as np
Delta = np.zeros((3200,141))
# for i in range(Delta.shape[0]):
#     Delta[i] = i

print(Delta.reshape(32,141,-1).shape)
