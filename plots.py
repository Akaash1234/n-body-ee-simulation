import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import os

def plot_orbits(history, names, skip=10):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    n_bodies = len(names)
    n_steps = len(history)
    
    for i in range(n_bodies):
        path = np.array([step['pos'][i] for step in history])
        ax.plot(path[::skip, 0], path[::skip, 1], path[::skip, 2], label=names[i])
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2])
        
    ax.legend()
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    plt.show()

def make_gif(history, names, filename='orbit.gif', skip=10):
    images = []
    n_bodies = len(names)
    
    for t in range(0, len(history), skip):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(n_bodies):
            path = np.array([step['pos'][i] for step in history[:t+1]])
            if len(path) > 0:
                ax.plot(path[:, 0], path[:, 1], path[:, 2], alpha=0.5)
                ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], label=names[i])
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        
        plt.savefig('temp.png')
        images.append(imageio.imread('temp.png'))
        plt.close()
        
    imageio.mimsave(filename, images)
    os.remove('temp.png')
