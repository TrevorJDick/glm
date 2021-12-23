# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:18:14 2021

@author: TD
"""
import numpy as np

from scipy.stats import t as students_t


def bootstrap_mean_var_y(model, x, y, sample_weights=None, x_test=None, 
                         nbootstraps=None, resample_frac=0.7):
    if x_test is None:
        x_test = x
    
    if nbootstraps is None:
        nbootstraps = x.shape[0]
    
    Y_test = []
    for nbs in range(nbootstraps):
        # sample with replacement x,y train data (bootstrapping)
        idx = np.random.choice(
            x.shape[0],
            int(resample_frac * x.shape[0]),
            replace=True
        )
        x_i, y_i = (x[idx], y[idx])
        
        # fit model
        model.fit(x_i, y_i, sample_weights=sample_weights)
        
        # predict on x_test
        y_test_i = model.predict(x_test)
        Y_test.append(y_test_i)
        
    Y_test = np.array(Y_test)
    
    y_test_fit = np.mean(Y_test, axis=0)
    var_f_test = np.std(Y_test, axis=0, ddof=1) ** 2 ## ddof=1 unbiased
    return y_test_fit, var_f_test


def t_crit(dof, significance_level=0.05):
    if significance_level is None:
        significance_level = 0.05 # default for 95%
    p = 1 - (significance_level / 2)
    t = students_t.ppf(p, dof)
    return t


def base_pred_conf_interval(var_y, dof, significance_level=None):
    interval = (
        t_crit(dof, significance_level=significance_level)
        * np.sqrt(var_y)
    )
    return interval


### TODO make class? so dont have to redo bootstrap_mean_var_y
# when switching between pred and conf intervals
def bootstrap_pred_conf_interval(model, x, y, sample_weights=None, x_test=None,
                                 nbootstraps=None, resample_frac=0.7,
                                 interval_type='pred', significance_level=None):
    y_test_fit, var_f_test = bootstrap_mean_var_y(
        model, 
        x, 
        y,
        sample_weights=sample_weights,
        x_test=x_test,
        nbootstraps=nbootstraps, 
        resample_frac=resample_frac
    )
    
    dof = y.shape[0]
    var_y = var_f_test.copy()
    # default is therefore a confidence interval when if is not triggered
    if interval_type in ('pred', 'prediction'):
        sigma_sqrd = np.sum((y - model.predict(x)) ** 2) / dof
        var_y += sigma_sqrd
    
    pred_conf_interval = base_pred_conf_interval(
        var_y,
        dof,
        significance_level=significance_level
    )
    return var_f_test, pred_conf_interval