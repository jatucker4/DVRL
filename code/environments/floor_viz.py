import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def plot_maze(states, figure_name):
    plt.figure(figure_name)
    ax = plt.axes()

    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    # goals
    if states[0, 1] <= 0.5:
        cir = plt.Circle((2, 0.25), 0.07, color='orange')
    else:
        cir = plt.Circle((0, 0.75), 0.07, color='orange')
    ax.add_artist(cir)

    walls = np.array([
        # horizontal
        [[0, 0], [2, 0]],
        [[0, 0.5], [2, 0.5]],
        [[0, 1], [2, 1]],
        # vertical
        [[0, 0], [0, 1]],
        [[2, 0], [2, 1]],
        [[0.4, 0.4], [0.4, 0.5]],
        [[1.2, 0.9], [1.2, 1]],
        [[0.4, 0.0], [0.4, 0.1]],
        [[1.2, 0.5], [1.2, 0.6]],
    ])
    walls_dotted = np.array([
        [[0, 0.4], [2, 0.4]],
        [[0, 0.9], [2, 0.9]],
        [[0, 0.6], [2, 0.6]],
        [[0, 0.1], [2, 0.1]],
    ])

    color = (0, 0, 0)
    ax.plot(walls[:, :, 0].T, walls[:, :, 1].T, color=color, linewidth=1.0)

    color = (0, 0, 1)
    ax.plot(walls_dotted[:, :, 0].T, walls_dotted[:, :, 1].T, color=color, linewidth=1.0, linestyle='--')

    if type(states) is np.ndarray:
        xy = states[:,:2]
        x, y = zip(*xy)
        ax.plot(x[0], y[0], 'bo')
        # Iterate through x and y with a colormap
        colorvec = np.linspace(0, 1, len(x))
        viridis = cm.get_cmap('YlGnBu', len(colorvec))
        for i in range(len(x)):
            if i == 0:
                continue
            plt.plot(x[i], y[i], color=viridis(colorvec[i]), marker='o')

    ax.set_aspect('equal')
    plt.savefig(figure_name)
    plt.close()