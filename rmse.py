import numpy as np
from sys import argv
f1n, f2n = argv[1:3]
f1 = np.array(open(f1n,'r').readlines()).astype(np.float)
f2 = np.array(open(f2n,'r').readlines()).astype(np.float)
print(f1.shape,f2.shape)
print("RMSQ", np.sqrt(np.mean((f1 - f2) ** 2)))