import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
state = np.zeros((100, 6))
# formation = np.array([[0, 0],
#                       [1, 0],
#                       [0, 1]])
# [0,0,1,0,x,y]
state[:, 2] = 1
state[:, 4] = x

a = torch.load('test.pt')
for i in range(100):
    state[:, 5] = y[i]
    s = torch.FloatTensor(state)
    action_value = a(s)
    max_value, action = torch.max(action_value, dim=1)
    Z[i, :] = max_value.detach().numpy()

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

U = np.zeros_like(Z)



plt.show()
