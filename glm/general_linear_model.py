# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:58:38 2021

@author: TD
"""
import numpy as np


class GLM:
    
    def __init__(self, basis_funcs=([lambda x: np.ones_like(x), lambda x:x],)):
        """
        Parameters
        ----------
        basis_funcs : list or tuple
            List of lists of functions, where each list in the list is
            the basis functions corresponding to a single dimension.  
            
            For example, if passed
            ([lambda x: np.ones_like(x), lambda x: x], [lambda x: x])
            this is equivalent to the basis {1, x, y} for the equation
            z = Ax + By + C; an equation for a plane.
            
            The default is the basis for an equation of a line
            in 1-dimension ([lambda x: 1, lambda x:x],).
        """
        self.basis_funcs = basis_funcs
    
    
    def fit(self, X, y, sample_weights=None):
        self.domains = GLM.domains_from_X(X) # used for linear indep checks
        A = GLM.create_design_matrix(self.basis_funcs, X)
        W = GLM.get_sample_weight_matrix(sample_weights, y)
        
        try:
            B = GLM.compute_b_matrix(A, W)
            self.beta = np.dot(B, y)
        
            ### for diagnotics and stats 
            # trace(W.M)
            self.dof = np.trace(
                np.dot(
                    W,
                    GLM.compute_m_matrix(
                        GLM.compute_projection_matrix(A, B)
                    )
                )
            )
            self.sigma_sqrd = GLM.compute_sigma_sqrd(
                y - GLM._func_glm(A, self.beta), 
                W,
                self.dof
            )
            self.var_beta = GLM.compute_var_beta(self.sigma_sqrd, B)
        except Exception as e:
           self._handle_fit_error(e)
        return self
    
    
    def predict(self, X):
        ### address trade off here:
        # must recompute design matrix each predict call vs
        # not saving design matrix during fit reduces memory footprint
        return GLM.func_glm(self.basis_funcs, self.beta, X)
    
    
    @staticmethod
    def create_design_matrix(basis_funcs, X):
        A = np.concatenate(
            [
                np.array(
                    [f(X[:, i]) for f in bf_coord]
                ).T
                for i, bf_coord in enumerate(basis_funcs)
            ],
            axis=1
        )
        return A
    
    
    @staticmethod
    def _func_glm(A, beta):
        return np.dot(A, beta)
    
    
    @staticmethod
    def func_glm(basis_funcs, beta, X):
        A = GLM.create_design_matrix(basis_funcs, X)
        return GLM._func_glm(A, beta)
    
    
    @staticmethod
    def jac_glm(basis_funcs, X):
        return GLM.create_design_matrix(basis_funcs, X)
    
    
    @staticmethod
    def jac_objective(basis_funcs, beta, X, y, sample_weights=None):
        A = GLM.create_design_matrix(basis_funcs, X)
        e = y - np.dot(A, beta) # faster not to call func_glm
        W = GLM.get_sample_weight_matrix(sample_weights, y)
        return 2 * np.dot(np.dot(e.T, W), A)
    
    
    @staticmethod
    def get_sample_weight_matrix(sample_weights, y):
        if sample_weights is None:
            sample_weights = np.identity(y.shape[0])
        
        if not isinstance(sample_weights, np.ndarray):
            sample_weights = np.array(sample_weights)
        
        W = sample_weights
        if len(W.shape) < 2:
            W = np.diag(W)
            
        if len(W.shape) != 2:
            raise ValueError(
                f'{W.shape} -- weights matrix shape. 2-d matrix required!'
            )
        if (W.shape[0] != W.shape[1]):
            raise ValueError(
                f'{W.shape} -- weights matrix shape. Matrix is not square!'
            )
        if (W.shape[0] != len(y)):
            raise ValueError(
                f'{W.shape} -- weights matrix shape.\n'
                f'{len(y)} -- number of samples.\n'
                'Weight matrix should have shape nsamples x nsamples!'
            )
        return W
    
    
    @staticmethod
    def compute_b_matrix(A, W):
        """
        beta = B.y
        """
        ATW = np.dot(A.T, W)
        
        V = np.dot(ATW, A)
        if np.linalg.det(V) == 0:
            raise ValueError(
                'NO SOLUTION: det(A^T.W.A)=0\n'
                'Check design matrix for linear independence, '
                'or sample weight matrix contribution to A^T.W.A!'
            )
        B = np.dot(np.linalg.inv(V), ATW)
        del V, ATW
        return B
    
    
    @staticmethod
    def compute_projection_matrix(A, B):
        """
        projection matrix is idempotent P^2 = P
        
        y_fit = P.y
        """
        return np.dot(A, B)
    
    
    @staticmethod
    def compute_m_matrix(PA):
        """
        residual operator matrix
        
        e = M.y
        """
        return np.identity(PA.shape[0]) - PA
    
    
    @staticmethod
    def compute_sigma_sqrd(e, W, dof):
        return  np.dot(np.dot(e.T, W), e) / dof
    
    
    @staticmethod
    def compute_var_beta(sigma_sqrd, B):
        return sigma_sqrd * np.dot(B, B.T)
    
    
    @staticmethod
    def basis_funcs_slice_indices(basis_funcs):
        # use if ever need to slice cols of A when basis is multidimensional
        # prepend the zero length for index algorithm purposes
        basis_lens = [0] + [len(i) for i in basis_funcs]
        slice_indices = np.array(
            [
                np.cumsum(basis_lens[:-1]), 
                np.cumsum(basis_lens[1:])
            ]
        ).T
        return slice_indices
    
    
    @staticmethod
    def domains_from_X(X):
        domains = np.array([np.min(X, axis=0), np.max(X, axis=0)]).T
        return domains
    
    
    @staticmethod
    def check_linear_independence_basis_funcs(basis_funcs, domains):
        step = 0.01
        lin_indep_pass = []
        ranks = []
        for bf, domain in zip(basis_funcs, domains):
            m = int((domain[1] - domain[0]) / step)
            n = len(bf)
            r = np.linalg.matrix_rank(
                GLM.create_design_matrix(
                    [bf],
                    np.linspace(*domain, m).reshape(-1, 1)
                )
            )
            ranks.append(r)
            if r == n:
                lin_indep_pass.append(True)
            else:
                lin_indep_pass.append(False)
        return lin_indep_pass, ranks
    
    
    @staticmethod
    def linear_independence_check_print(basis_funcs, lin_indep_pass, ranks):
        for i, (bf, p, r) in enumerate(zip(basis_funcs, lin_indep_pass, ranks)):
            basis_message = 'TRUE -- basis function are linearly independent'
            if not p:
                basis_message =  'FALSE -- basis function are NOT linearly independent!'
            print(
                f'{i + 1} -- coordinate dimension:\n'
                f'{basis_message}\n'
                f'{r} -- rank of design matrix\n'
                f'{len(bf)} -- expected rank.\n'
            )
        print(
            'NOTE: once linear independance is re-established it is best to also '
            'check the data in X, y is reasonably filling up the space it is '
            'contained in.\n'
        )
        return
    
    
    def _handle_fit_error(self, exception):
        print(exception)
        lin_indep_pass, ranks = GLM.check_linear_independence_basis_funcs(
            self.basis_funcs,
            self.domains
        )
        if all(lin_indep_pass):
            print(
                'All basis functions, for each dimension, are linearly '
                'independent.  The failure of finding an inverse for '
                'A^T.W.A is likely due to the behavoir of the higher '
                'dimensional data in X, y actually behaving as lower '
                'dimensional data.  An example is a 1D line embedded in 3D. '
                'The alternate possibility is the nature of sample weights, '
                'if they were provided.'
            )
        else:
            GLM.linear_independence_check_print(
                self.basis_funcs,
                lin_indep_pass,
                ranks
            )
        return