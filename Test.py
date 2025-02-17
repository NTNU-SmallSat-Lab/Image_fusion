from spatial_transform import find_overlap
import numpy as np
import matplotlib.pyplot as plt

limits = [0, 10, 0, 10]
limits_2 = [2, 5, 4, 12]
transform = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

mask = find_overlap(limits, limits_2, transform)

plt.imshow(mask)
plt.show()