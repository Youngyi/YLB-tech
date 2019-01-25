import numpy as np



arr6 = np.empty((8,4))
for i in range(8):#给每一行赋值
    arr6[i] = i

print(arr6.reshape([1,-1]))