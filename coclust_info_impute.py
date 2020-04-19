import numpy as np
import pandas as pd
import scipy.sparse as sp
from coclust.io.input_checking import check_positive
from coclust.initialization import random_init
from sklearn.utils import check_random_state, check_array
from utils import (summarize_blocks, get_block_counts, _impute_block_representative)
from scipy.sparse.sputils import isdense
from coclust.coclustering.base_non_diagonal_coclust import BaseNonDiagonalCoclust


class CoclustInfoImpute(BaseNonDiagonalCoclust):
    """Information-Theoretic Co-clustering.
    Parameters
    ----------
    n_row_clusters : int, optional, default: 2
        Number of row clusters to form
    n_col_clusters : int, optional, default: 2
        Number of column clusters to form
    init : numpy array or scipy sparse matrix, \
        shape (n_features, n_clusters), optional, default: None
        Initial column labels
    max_iter : int, optional, default: 20
        Maximum number of iterations
    n_init : int, optional, default: 1
        Number of time the algorithm will be run with different
        initializations. The final results will be the best output of `n_init`
        consecutive runs.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    tol : float, default: 1e-9
        Relative tolerance with regards to criterion to declare convergence
    Attributes
    ----------
    row_labels_ : array-like, shape (n_rows,)
        Bicluster label of each row
    column_labels_ : array-like, shape (n_cols,)
        Bicluster label of each column
    delta_kl_ : array-like, shape (k,l)
        Value :math:`\\frac{p_{kl}}{p_{k.} \\times p_{.l}}` for each row
        cluster k and column cluster l
    """

    def __init__(self, n_row_clusters=2, n_col_clusters=2, init=None,
                 max_iter=20, n_init=1, tol=1e-9, random_state=None):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.row_labels_ = None
        self.column_labels_ = None
        self.criterions = []
        self.criterion = -np.inf
        self.delta_kl_ = None
        
    def initial_impute(self,data):
        """ Make a initial imputation 
        """
        X = data.copy()
        na_rows, na_cols = np.where(np.isnan(X))
        X[na_rows, na_cols] = 0
        return X, na_rows, na_cols
    
    def fit(self, X,impute_func=None,na_rows=None,na_cols=None, y=None):
        """Perform co-clustering.
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        
        X_ = X.copy()
        
        if np.isnan(X_).sum() > 0:
            X_, na_rows, na_cols = self.initial_impute(X_)

        random_state = check_random_state(self.random_state)

        check_array(X_, accept_sparse=True, dtype="numeric", order=None,
                    copy=False, force_all_finite=True, ensure_2d=True,
                    allow_nd=False, ensure_min_samples=self.n_row_clusters,
                    ensure_min_features=self.n_col_clusters,
                    warn_on_dtype=False, estimator=None)

        check_positive(X_)

        X_ = X_.astype(float)
        
        criterion = self.criterion
        criterions = self.criterions
        row_labels_ = self.row_labels_
        column_labels_ = self.column_labels_
        delta_kl_ = self.delta_kl_
        best_X = X_

        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            self._fit_single(X_,impute_func,na_rows, na_cols, seed, y)
            if np.isnan(self.criterion):
                raise ValueError("matrix may contain negative or "
                                 "unexpected NaN values")
            # remember attributes corresponding to the best criterion
            if (self.criterion > criterion):
                criterion = self.criterion
                criterions = self.criterions
                row_labels_ = self.row_labels_
                column_labels_ = self.column_labels_
                delta_kl_ = self.delta_kl_
                best_X = self.X_

        # update attributes
        self.criterion = criterion
        self.criterions = criterions
        self.row_labels_ = row_labels_
        self.column_labels_ = column_labels_
        self.delta_kl_ = delta_kl_
        self.X_ = best_X

        return self

    def _fit_single(self, X,impute_func,na_rows, na_cols, random_state, y=None):
        """Perform one run of co-clustering.
        Parameters
        ----------
        X : numpy array or scipy sparse matrix, shape=(n_samples, n_features)
            Matrix to be analyzed
        """
        
        K = self.n_row_clusters
        L = self.n_col_clusters
        
        if self.init is None:
            W = random_init(L, X.shape[1], random_state)
        else:
            W = np.matrix(self.init, dtype=float)
        w_ = np.argmax(W, axis=1)[:, np.newaxis]
        X = sp.csr_matrix(X)

        N = float(X.sum())
        X = X.multiply(1. / N)

        Z = sp.lil_matrix(random_init(K, X.shape[0], self.random_state))

        W = sp.csr_matrix(W)

        # Initial delta
        p_il = X * W
        # p_il = p_il     # matrix i,l ; column l' contains the p_il'
        p_kj = X.T * Z  # matrix j,k

        p_kd = p_kj.sum(axis=0)  # array containing the p_k.
        p_dl = p_il.sum(axis=0)  # array containing the p_.l

        # p_k. p_.l ; transpose because p_kd is "horizontal"
        p_kd_times_p_dl = p_kd.T * p_dl
        min_p_kd_times_p_dl = np.nanmin(
            p_kd_times_p_dl[
                np.nonzero(p_kd_times_p_dl)])
        p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
        p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl

        p_kl = (Z.T * X) * W
        delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)

        change = True
        news = []
        
        n_iters = self.max_iter
        pkl_mi_previous = float(-np.inf)

        # Loop
        while change and n_iters > 0:
            change = False
            
            # Update Z
            p_il = X * W  # CSR matrix i,l 
            if not isdense(delta_kl):
                delta_kl = delta_kl.todense()
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl.T)
            log_delta_kl = sp.csr_matrix(log_delta_kl)

            # p_il matmul (log d_kl)T ; we examine each cluster
            Z1 = p_il @ log_delta_kl
            # Z1 is CSR which is bad for support item assignment. So convert Z1
            Z1 = Z1.toarray()
            Z = np.zeros_like(Z1)
            # Z[(line index 1...), (max col index for 1...)]
            Z[np.arange(len(Z1)), Z1.argmax(1)] = 1

            # impute missing value
            z_ = np.argmax(Z, axis=1)[:, np.newaxis]
            X = impute_func(X, Z, W.toarray(), z_, w_, na_rows, na_cols)
            Z = sp.csr_matrix(Z)
            
            # Update delta
            # matrice d, k ; column k' contains the p_jk'
            p_kj = X.T * Z
            # p_il unchanged so no need for p_dl = p_il.sum(axis=0) 
            p_kd = p_kj.sum(axis=0)  # array k containing the p_k.

            # p_k. p_.l ; transpose because p_kd is "horizontal"
            p_kd_times_p_dl = p_kd.T * p_dl
            min_p_kd_times_p_dl = np.nanmin(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W
            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)

            # Update W
            p_kj = X.T * Z  # CSR matrice j,k
            if not isdense(delta_kl): # delta_kl should be a sparse coo here
                delta_kl = delta_kl.todense()
            delta_kl[delta_kl == 0.] = 0.0001  # to prevent log(0)
            log_delta_kl = np.log(delta_kl)
            log_delta_kl = sp.lil_matrix(log_delta_kl)
            W1 = p_kj * log_delta_kl  # p_kj * d_kl ; we examine each cluster
            # W1 is CSR which is bad for support item assignment. So convert W1
            W1 = W1.toarray()
            W = np.zeros_like(W1)
            W[np.arange(len(W1)), W1.argmax(1)] = 1
            # impute missing value
            w_ = np.argmax(W, axis=1)[:, np.newaxis]
            X = impute_func(X, Z.toarray(), W, z_, w_, na_rows, na_cols)
            W = sp.csr_matrix(W)

            # Update delta
            p_il = X * W     # matrix d,k ; column k' contains the p_jk'
            # p_kj unchanged so no need for p_kd = p_kj.sum(axis=0) 
            p_dl = p_il.sum(axis=0)  # array l containing the p_.l
            

            # p_k. p_.l ; transpose because p_kd is "horizontal"
            p_kd_times_p_dl = p_kd.T * p_dl
            min_p_kd_times_p_dl = np.nanmin(
                p_kd_times_p_dl[
                    np.nonzero(p_kd_times_p_dl)])
            p_kd_times_p_dl[p_kd_times_p_dl == 0.] = min_p_kd_times_p_dl * 0.01
            p_kd_times_p_dl_inv = 1. / p_kd_times_p_dl
            p_kl = (Z.T * X) * W 

            delta_kl = p_kl.multiply(p_kd_times_p_dl_inv)
            # to prevent log(0) when computing criterion
            # but note that delta_kl = csr eletwise-mult array is a coo
            # which does not support item assignment
            if not isdense(delta_kl):
                delta_kl = delta_kl.todense()
            delta_kl[delta_kl == 0.] = 0.0001

            # Criterion
            # p_kl is csr_matrix; delta_kl is np.matrix
            # just in case, note that their matmul results in a COO matrix !
            pkl_mi = p_kl.multiply(np.log(delta_kl) )
            pkl_mi = pkl_mi.sum()

            if np.abs(pkl_mi - pkl_mi_previous) > self.tol:
                pkl_mi_previous = pkl_mi
                change = True
                news.append(pkl_mi)
                n_iters -= 1

        self.criterions = news         
        self.criterion = pkl_mi
        self.row_labels_ = Z.toarray().argmax(axis=1).tolist()
        self.column_labels_ = W.toarray().argmax(axis=1).tolist()
        self.delta_kl_ = delta_kl
        self.Z = Z
        self.W = W
        self.X_ = X
    