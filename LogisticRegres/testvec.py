import numpy as np

a = np.array([1, 2, 3, 4])

b = np.array([4, 3, 2, 1])
c = np.array([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
])
a_ = a.transpose()*b
b_ = a*b.transpose()
print(a.shape, b.shape)
print(a_, b_)
print(a_.shape, b.shape)

print(c*a.transpose())
