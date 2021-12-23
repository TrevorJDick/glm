# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:58:38 2021

@author: TD
"""
import numpy as np


class GLM:
    
    def __init__(self, basis_funcs=([lambda x: 1, lambda x:x],)):
        self.basis_funcs = basis_funcs
    
    
    def fit(self, X, y, sample_weights=None):
        A = GLM.create_design_matrix(self.basis_funcs, X)
        W = GLM.get_sample_weight_matrix(sample_weights, y)
        ATW = np.dot(A.T, W)
        
        V = np.dot(ATW, A)
        if np.linalg.det(V) == 0:
            raise ValueError(
                'NO SOLUTION: det(A^T.W.A)=0\n'
                'Check design matrix or sample weight matrix!'
            )
        B = np.dot(np.linalg.inv(V), ATW)
        del V, ATW
        
        self.beta = np.dot(
            B,
            y
        )
        
        # for diagnotics and stats 
        PA = np.dot(A, B)
        del A
        M = np.identity(PA.shape[0]) - PA
        self.dof = np.trace(np.dot(W, M))
        del PA, M
        e = y - self.predict(X)
        self.sigma_sqrd = np.dot(np.dot(e.T, W), e) / self.dof
        del e, W
        self.var_beta = self.sigma_sqrd * np.dot(B, B.T)
        del B
        return self
    
    
    def predict(self, X):
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
    def func_glm(basis_funcs, beta, X):
        A = GLM.create_design_matrix(basis_funcs, X)
        return np.dot(A, beta)
    
    
    @staticmethod
    def jac_glm(basis_funcs, X):
        return GLM.create_design_matrix(basis_funcs, X)
    
    
    @staticmethod
    def jac_objective(basis_funcs, beta, X, y, sample_weights=None):
        A = GLM.create_design_matrix(basis_funcs, X)
        e = y - np.dot(A, beta)
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