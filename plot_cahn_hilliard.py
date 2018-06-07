import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("measurement_data\\f_sim2.txt")

plt.plot(data[:, 0], data[:, 1])
plt.ylabel('E')
plt.xlabel('t')
plt.show()
