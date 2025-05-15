import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1) Define weights and biases for a 3‐class softmax in 2D
W = np.array([
    [ 1.0,  0.0],               # Class 0 hyperplane normal
    [-0.5,  np.sqrt(3)/2],      # Class 1 hyperplane normal
    [-0.5, -np.sqrt(3)/2]       # Class 2 hyperplane normal
])
b = np.zeros(3)  # zero biases

# 2) Create a grid of (x,y) points over [-1,1]×[-1,1]
grid_size = 400
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
xx, yy = np.meshgrid(x, y)
points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # shape (N,2)

# 3) Compute logits and softmax probabilities
logits = points.dot(W.T) + b           # shape (N,3)
exp_logits = np.exp(logits)            # elementwise exp
probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # shape (N,3)

# 4) Determine class by argmax of the softmax
regions = np.argmax(probs, axis=1).reshape(xx.shape)

# 5) Plot the decision regions
plt.figure(figsize=(6,6))
cmap = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
plt.contourf(xx, yy, regions, levels=[-0.5,0.5,1.5,2.5], cmap=cmap, alpha=0.5)
plt.title("3-Class Softmax Decision Regions in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

