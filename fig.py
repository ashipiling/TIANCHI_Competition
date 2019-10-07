import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0.0, 2 * np.pi, 20 )
print(theta)
height = 10 * np.random.rand(20 )
width = np.pi / 4 * np.random.rand(20 )

ax = plt.subplot(111, projection='polar')
#polar 极圈图
bars = ax.bar(theta, height, width)



plt.show()