m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
# print(m.items())
adict = dict([(y, x) for (x, y) in m.items()])
print(adict)
import numpy as np

seq = [[1, 2, 3, 4, 5], [2, 3, 5, 5, 1], [2, 4, 0, 3, 1]]


np_result = np.zeros((5, 10))
# print(np_result)
for i,j in enumerate(seq):
    print(j)
    np_result[i,j] = 1
#
print(np_result)
