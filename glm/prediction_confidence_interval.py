# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 19:28:45 2021

@author: TD
"""
import numpy as np

import base_pred_conf_interval as bpc


def glm_pred_conf_interval(model, x, y, sample_weights=None, x_test=None,
                           interval_type='pred', significance_level=None):
    if x_test is None:
        x_test = x
    
    var_beta = model.var_beta
    dof = model.dof
    var_f = np.sum(
        np.dot(
            model.jac_glm(model.basis_funcs, x_test),
            var_beta
        ) * model.jac_glm(model.basis_funcs, x_test),
        axis=1
    )
    
    var_y = var_f.copy()
    # default is therefore a confidence interval when if is not triggered
    if interval_type in ('pred', 'prediction'):
        sigma_sqrd = model.sigma_sqrd
        var_y += sigma_sqrd
    
    pred_conf_interval = bpc.base_pred_conf_interval(
        var_y,
        dof,
        significance_level=significance_level
    )
    return var_f, pred_conf_interval