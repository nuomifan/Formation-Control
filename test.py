import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

xs = np.linspace(0, 100, 51)
ys = np.linspace(0, 100, 51)
ax = plt.gca()
# grid "shades" (boxes)
w, h = xs[1] - xs[0], ys[1] - ys[0]
for i, x in enumerate(xs[:-1]):
    for j, y in enumerate(ys[:-1]):
        if i % 2 == j % 2: # racing flag style
            ax.add_patch(Rectangle((x, y), w, h, fill=True, color='#008610', alpha=.1))
# grid lines
for x in xs:
    plt.plot([x, x], [ys[0], ys[-1]], color='black', alpha=.33, linestyle=':')
for y in ys:
    plt.plot([xs[0], xs[-1]], [y, y], color='black', alpha=.33, linestyle=':')
# plt.axis('equal')
plt.show()
