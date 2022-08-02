import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_batch(directoryPath):
    j = 0
    ax = plt.axes()
    for file_name in glob.glob(directoryPath+'*.csv'):
        glued_data = pd.DataFrame()
        x = pd.read_csv(file_name, low_memory=False,skiprows=1, delim_whitespace=True)
        glued_data = pd.concat([glued_data,x],axis=1)
        fig_name = "batch" + str(j) + ".png"

        x_vals = pd.DataFrame()
        y_vals = pd.DataFrame()
        rew_vals = pd.DataFrame()
        x_vals = glued_data.iloc[:, 0]
        y_vals = glued_data.iloc[:, 1]
        rew_vals = glued_data.iloc[:,2]
        
        
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 1])

        # goals
        
        cir1 = plt.Circle((2, 0.25), 0.07, color='orange')
        ax.add_artist(cir1)
        cir2 = plt.Circle((0, 0.75), 0.07, color='orange')
        ax.add_artist(cir2)

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

        c_vec = []
        for i in range(len(rew_vals.to_numpy())):
            c_vec.append([rew_vals.to_numpy()[i], rew_vals.to_numpy()[i], rew_vals.to_numpy()[i]])
        print(rew_vals.to_numpy())
        ax.scatter(x_vals.to_numpy(),y_vals.to_numpy(), color=c_vec)
        # plt.savefig(fig_name)
        j += 1


if __name__ == "__main__":
    directory = "/home/jtucker/DVRL_baseline/tmp/gym/57/tmp/gym/tmp/gym/13/"
    plot_batch(directory)
