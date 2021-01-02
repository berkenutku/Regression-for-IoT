import numpy as np

filters = [24, 36, 48, 64, 96, 128, 256, 384, 512]
kernels = [3, 5, 7, 11]
final = []
arrangement = []

for f in filters:
    for k in kernels:
        final.append(f*k)


print(final)
final = np.sort(final)
print(final)


        
