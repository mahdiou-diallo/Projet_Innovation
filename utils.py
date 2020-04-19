import matplotlib.pyplot as plt
import numpy as np


def show_matrix(X, no_axes=True):
    # ax = plt.imshow(X, cmap='Greys', vmin=0, vmax=X.max())
    ax = plt.spy(X)
    if no_axes:
        plt.axis('off')
    return ax

def RMSE(X1, X2):
    diff = np.abs(X1 - X2)
    r = ((diff**2).sum()/X2.shape[0])**.5
    return r

def summarize_blocks(X, z, wT):
    """get the summary matrix from a contingency matrix X
        with row labels z and column labels w
    Parameters
    ----------
    X: np.array, n x d, contingency table (n rows d, columns)
    z: np.array, n x k, row assignments (n rows, k classes)
    wT: np.array, d x l, transpose of column assignments (d columns, l classes)

    Returns
    -------
    for each block i,j return the sum of values in that block, shape k x l
    """
    return z.T @ X @ wT


def get_block_counts(z, wT):
    """get the count of item in each block of matrix X (n x d)
    Parameters
    ----------
    z: np.array, n x k, row assignments (n rows, k classes)
    wT: np.array, d x l, transpose of column assignments (d columns, l classes)

    Returns
    -------
    for each block i,j return the number of items in that block, shape k x l
    """
    
    return z.sum(axis=0)[:, np.newaxis] * wT.sum(axis=0)


def _impute_block_representative(X, Z, W, z, w, r_nan, c_nan):
    s = summarize_blocks(X, Z, W)
    bc = get_block_counts(Z, W)
    bc[bc == 0] = 1  # avoid divide by 0
    block_rep_vals = s / bc
    X[r_nan, c_nan] = block_rep_vals[z[r_nan].ravel(), w[c_nan].ravel()]
    return X