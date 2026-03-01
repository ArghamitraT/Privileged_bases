# import numpy as np
# import matplotlib.pyplot as plt

# # Create a grid in 2D
# xs = np.linspace(-2, 2, 400)
# ys = np.linspace(-2, 2, 400)
# X, Y = np.meshgrid(xs, ys)
# points = np.stack([X.ravel(), Y.ravel()], axis=1)

# # Define weight vector
# w = np.array([1.0, 0.0])  # vertical boundary x=0

# # ReLU
# def relu(z):
#     return np.maximum(0, z)

# # Rotation matrix (45 degrees)
# theta = np.pi / 4
# R = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta),  np.cos(theta)]
# ])

# # Case 1: Original ReLU
# Z1 = relu(points @ w)

# # Case 2: Rotate input before ReLU
# rot_points = points @ R.T
# Z2 = relu(rot_points @ w)

# # Case 3: Rotate output after ReLU
# Z3 = (R @ np.vstack([Z1, np.zeros_like(Z1)])).T[:, 0]

# # Plot
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# for ax, Z, title in zip(
#     axs,
#     [Z1, Z2, Z3],
#     ["Original ReLU", "Rotate input → ReLU", "ReLU → Rotate output"]
# ):
#     sc = ax.scatter(points[:, 0], points[:, 1], c=Z, cmap="coolwarm", s=1)
#     ax.set_title(title)
#     ax.set_aspect("equal")
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Grid
xs = np.linspace(-2, 2, 400)
ys = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(xs, ys)
pts = np.stack([X.ravel(), Y.ravel()], axis=1)

# Two ReLU neurons
W = np.array([
    [1.0, 0.0],   # vertical cut
    [0.0, 1.0],   # horizontal cut
])

def relu(z):
    return np.maximum(0, z)

# Rotation
theta = np.pi / 4
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Case 1: Original ReLU layer
Z1 = relu(pts @ W.T)

# Case 2: Rotate input -> ReLU
Z2 = relu((pts @ R.T) @ W.T)

# Case 3: ReLU -> rotate output
Z3 = (R @ Z1.T).T

# Visualize magnitude
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Original ReLU layer",
          "Rotate input → ReLU",
          "ReLU → Rotate output"]

for ax, Z, title in zip(axs, [Z1, Z2, Z3], titles):
    mag = np.linalg.norm(Z, axis=1)
    ax.scatter(pts[:, 0], pts[:, 1], c=mag, cmap="coolwarm", s=1)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.show()

