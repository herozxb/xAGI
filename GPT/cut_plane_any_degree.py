import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1) Create a grid over (-1,1)×(-1,1)
x = np.linspace(-1, 1, 200)
y = np.linspace(-1, 1, 200)
X, Y = np.meshgrid(x, y)

# 2) First‐layer ReLUs
h1 = np.maximum(0, X)   # ReLU(x)
h2 = np.maximum(0, Y)   # ReLU(y)

theta = 45.0 / 180.0 * 3.1415926

# 3) Second‐layer linear combination
f = np.cos(theta) * h1 + np.sin(theta) * h2            # Defines the 45° plane (f=0 along y=x)

# 4) Plot each surface
def plot_surface(Z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(title.split('=')[0].strip())
    ax.view_init(elev=30, azim=-60)

plot_surface(h1, 'h1 = ReLU(x)')
plot_surface(h2, 'h2 = ReLU(y)')
plot_surface(f,  'f  = h1 + h2')

plt.show()

