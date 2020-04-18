import numpy as np
from sklearn.preprocessing import normalize


def random_cells(X, n_out):
    """Given an n x r matrix X, returns n_out distinct cell positions
    Parameters
    ----------
    X: np.ndarray or np.matrix, a 2D matrix
    n_out: int, the number of cells to select.
    """
    n, nc = X.size, X.shape[1]
    choices = np.random.choice(n, size=n_out, replace=False)
    rows = choices // nc
    cols = choices % nc
    return rows, cols


def ampute_mcar(X, prop=.2, random_state=42):
    np.random.seed(random_state)
    n_missing = int(X.size * prop)
    r_nan, c_nan = random_cells(X, n_missing)
    res = X.astype(float)
    res[r_nan, c_nan] = np.nan
    return res, r_nan, c_nan


def compute_mar_probas(X_complete, W=None, random_state=42):
    np.random.seed(random_state)
    X_obs = X_complete.copy().astype(float)
    M_proba = np.zeros(X_obs.shape)

    if W is None:
        # generate the weigth matrix W
        W = np.random.randn(X_complete.shape[1], X_complete.shape[1])

    # Severals iteration to have room for high missing_rate
    for i in range(min(20, X_obs.shape[1])):
        # Sample a pattern matrix P
        # P[i,j] = 1 will correspond to an observed value
        # P[i,j] = 0 will correspond to a potential missing value
        P = np.random.binomial(1, .5, size=X_complete.shape)

        # potential missing entry do not take part of missingness computation
        X_not_missing = np.multiply(X_complete, P)

        # sample from the proba X_obs.dot(W)
        sigma = np.var(X_not_missing)
        M_proba_ = np.random.normal(X_not_missing.dot(W), scale=sigma)

        # not missing should have M_proba = 0
        M_proba_ = np.multiply(M_proba_, 1-P)  # M_proba[P] = 0

        M_proba += M_proba_

    return M_proba, X_obs


def ampute_mar(X_complete, prop=.2, M_proba=None, W=None, random_state=42):
    """ Observed values will censor the missing ones

    The proba of being missing: M_proba = X_obs.dot(W)
    So for each sample, some observed feature (P=1) will influence 
    the missingness of some others features (P=0) w.r.t to the weight 
    matrix W (shape n_features x n_features).

    e.g. during a questionnary, those who said being busy (X_obs[:,0] = 1) 
    usualy miss to fill the last question (X_obs[:,-1] = np.nan)
    So here W[0,-1] = 1
    [source](https://rmisstastic.netlify.com/how-to/python/generate_html/how%20to%20generate%20missing%20values)
    """
    if M_proba is None:
        M_proba, X_obs = compute_mar_probas(
            X_complete, W=W, random_state=random_state)
    else:
        X_obs = X_complete.astype(float).copy()

    thresold = np.percentile(M_proba.ravel(), 100 * (1 - prop))
    M = M_proba > thresold

    np.putmask(X_obs, M, np.nan)
    print('Percentage of newly generated mising values: {}'.
          format(np.sum(np.isnan(X_obs))/X_obs.size))
    return (X_obs, *np.where(M))
