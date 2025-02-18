import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import numpy as np
import pickle

# from stanford_client import StanfordEnvironmentClient
# env = StanfordEnvironmentClient()
      

def plot_maze(episode, figure_name, test_traps=None):
    from stanford_client import StanfordEnvironmentClient
    env = StanfordEnvironmentClient()

    xlim = env.xrange
    ylim = env.yrange
    goal = [env.target_x[0], env.target_y[0], 
            env.target_x[1]-env.target_x[0], env.target_y[1]-env.target_y[0]]
    trap1_x = env.trap_x[0]
    trap2_x = env.trap_x[1]
    trap1 = [trap1_x[0], env.trap_y[0], 
            trap1_x[1]-trap1_x[0], env.trap_y[1]-env.trap_y[0]]
    trap2 = [trap2_x[0], env.trap_y[0], 
            trap2_x[1]-trap2_x[0], env.trap_y[1]-env.trap_y[0]]
    dark = [env.xrange[0], env.yrange[0], env.xrange[1]-env.xrange[0], env.dark_line-env.yrange[0]]

    states = episode['state']

    plt.figure()
    ax = plt.axes()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # goal: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((goal[0], goal[1]), goal[2], goal[3], facecolor='green'))
    # trap i: [start_x, start_y, width, height]
    ax.add_patch(Rectangle((trap1[0], trap1[1]), trap1[2], trap1[3], facecolor='orange'))
    ax.add_patch(Rectangle((trap2[0], trap2[1]), trap2[2], trap2[3], facecolor='orange'))
    # dark region
    ax.add_patch(Rectangle((dark[0], dark[1]), dark[2], dark[3], facecolor='black', alpha=0.15))
    # additional wall
    ax.add_patch(Rectangle((0, trap1[1]), 
        trap1[0], trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap1[0]+trap1[2], trap1[1]), 
        goal[0]-(trap1[0]+trap1[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((goal[0]+goal[2], trap1[1]), 
        trap2[0]-(goal[0]+goal[2]), trap1[3], facecolor='black', alpha=0.2))
    ax.add_patch(Rectangle((trap2[0]+trap2[2], trap1[1]), 
        xlim[1]-(trap2[0]+trap2[2]), trap1[3], facecolor='black', alpha=0.2))

    if test_traps is not None:
        # trap i: [start_x, start_y, width, height]
        test_trap1_x = test_traps[0] #env.test_trap_x[0]
        test_trap1_y = test_traps[1] #env.test_trap_y[0]
        test_trap2_x = test_traps[2] #env.test_trap_x[1]
        test_trap2_y = test_traps[3] #env.test_trap_y[1]
        test_trap1 = [test_trap1_x[0], test_trap1_y[0], 
                test_trap1_x[1]-test_trap1_x[0], test_trap1_y[1]-test_trap1_y[0]]
        test_trap2 = [test_trap2_x[0], test_trap2_y[0], 
                test_trap2_x[1]-test_trap2_x[0], test_trap2_y[1]-test_trap2_y[0]]
        
        ax.add_patch(Rectangle((test_trap1[0], test_trap1[1]), test_trap1[2], test_trap1[3], facecolor='orange'))
        ax.add_patch(Rectangle((test_trap2[0], test_trap2[1]), test_trap2[2], test_trap2[3], facecolor='orange'))

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
    plt.savefig(figure_name, bbox_inches='tight', dpi=1000)
    # plt.savefig(figure_name, bbox_inches='tight')
    plt.close()