# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 11:36:30 2021

@author: TD
"""
import numpy as np

from scipy.stats import t as students_t


def t_crit(dof, significance_level=None):
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
