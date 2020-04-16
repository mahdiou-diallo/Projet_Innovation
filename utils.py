import matplotlib.pyplot as plt
import numpy as np


def show_matrix(X, no_axes=True):
    # ax = plt.imshow(X, cmap='Greys', vmin=0, vmax=X.max())
    ax = plt.spy(X)
    if no_axes:
        plt.axis('off')
    return ax
