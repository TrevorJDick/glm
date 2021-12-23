# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:59:52 2021

@author: TD
"""
import matplotlib.pyplot as plt
import numpy as np

import bootstrapped_pred_conf_interval as bpci
import general_linear_model as glm
import prediction_confidence_intervals as pci




beta_actual = np.array([0.5, 2.28901, np.pi])
basis_funcs = ([lambda x:x, np.sin, np.tanh],)

n = 5000
X = np.random.uniform(low=-10, high=10, size=n).reshape((-1,1))
y = glm.GLM.func_glm(basis_funcs, beta_actual, X) + 20 * (2 * np.random.rand(X.shape[0],) - 1)

plt.scatter(X[:, 0], y, s=15)
plt.show()

model = glm.GLM(basis_funcs=basis_funcs)
model.fit(X, y, sample_weights=None)

n2 = 1000
x_fit = np.linspace(-10, 10, n2).reshape(-1, 1)
y_fit = model.predict(x_fit)

plt.scatter(X[:, 0], y, s=15)
plt.plot(x_fit[:, 0], y_fit, lw=2, color='red')
plt.show()


print(
    f'{model.dof} -- degrees of freedom\n'
    f'{model.sigma_sqrd} -- sigma squared\n'
    f'{model.beta} -- optimal beta\n'
    f'{model.var_beta}\n -- variance of beta\n'
)

interval_type = 'pred'
sigma_sqrd = model.sigma_sqrd

var_f, pred_conf_interval = pci.glm_pred_conf_interval(
    model,
    X, 
    y, 
    sample_weights=None,
    x_test=x_fit,
    interval_type=interval_type, 
    significance_level=None
)

plt.scatter(X[:, 0], y, s=15)
plt.plot(x_fit, y_fit, lw=2, color='red', label='fit')
plt.plot(x_fit, y_fit - pred_conf_interval, color='purple', linestyle='--', label=f'{interval_type} interval')
plt.plot(x_fit, y_fit + pred_conf_interval, color='purple', linestyle='--')
plt.legend()
plt.show()


plt.plot(x_fit, glm.GLM.func_glm(model.basis_funcs, beta_actual, x_fit), lw=2, color='green', label='actual')
plt.plot(x_fit, y_fit, lw=2, color='red', label='fit')
plt.plot(x_fit, y_fit - pred_conf_interval, color='purple', linestyle='--', label=f'{interval_type} interval')
plt.plot(x_fit, y_fit + pred_conf_interval, color='purple', linestyle='--')
plt.legend()
plt.show()




### bootstrapped
var_f_bs, pred_conf_interval_bs = bpci.bootstrap_pred_conf_interval(
    model,
    X, 
    y, 
    sample_weights=None,
    x_test=x_fit,
    nbootstraps=50,
    resample_frac=0.7,
    interval_type=interval_type, 
    significance_level=None
)

plt.scatter(X[:, 0], y, s=15)
plt.plot(x_fit, y_fit, lw=2, color='red', label='fit')
plt.plot(x_fit, y_fit - pred_conf_interval_bs, color='purple', linestyle='--', label=f'{interval_type} interval bootstrapped')
plt.plot(x_fit, y_fit + pred_conf_interval_bs, color='purple', linestyle='--')
plt.legend()
plt.show()


plt.plot(x_fit, glm.GLM.func_glm(model.basis_funcs, beta_actual, x_fit), lw=2, color='green', label='actual')
plt.plot(x_fit, y_fit, lw=2, color='red', label='fit')
plt.plot(x_fit, y_fit - pred_conf_interval, color='purple', linestyle='--', label=f'{interval_type} interval')
plt.plot(x_fit, y_fit + pred_conf_interval, color='purple', linestyle='--')
plt.plot(x_fit, y_fit - pred_conf_interval_bs, color='orange', linestyle='--', label=f'{interval_type} interval bootstrapped')
plt.plot(x_fit, y_fit + pred_conf_interval_bs, color='orange', linestyle='--')
plt.legend()
plt.show()


plt.scatter(var_f, var_f_bs, s=4)
plt.plot(
    np.linspace(0, max(var_f.max(), var_f_bs.max()), 100),
      np.linspace(0, max(var_f.max(), var_f_bs.max()), 100),
      linestyle='--', color='black', lw=2
  )
plt.show()



### 3D example ###
beta_actual = np.array([0.5, 2.28901, np.pi, -1.5])
basis_funcs = ([lambda x:x, np.sin, np.tanh], [lambda y: y**2])

n = 500
X = np.concatenate(
    [np.random.uniform(low=-10, high=10, size=n).reshape(-1,1)
     for i in range(len(basis_funcs))],
    axis=1
)
y = glm.GLM.func_glm(basis_funcs, beta_actual, X) + 20 * (2 * np.random.rand(X.shape[0],) - 1)

for i in range(len(basis_funcs)):
    plt.scatter(X[:, i], y, s=15)
    plt.show()
    del i
    

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, s=50)
plt.show()
del ax, fig


model = glm.GLM(basis_funcs=basis_funcs)
model.fit(X, y, sample_weights=None)

n2 = 1000
x_fit =  np.concatenate(
    [np.linspace(-10, 10, n2).reshape(-1, 1)
     for i in range(len(basis_funcs))],
    axis=1
)
y_fit = model.predict(x_fit)


xx1, xx2 = np.meshgrid(x_fit[:, 0], x_fit[:, 1])
xx = np.concatenate(
    [xx1.reshape(*xx1.shape, 1), xx2.reshape(*xx1.shape, 1)],
    axis=2
).reshape(xx1.shape[0] * xx1.shape[1], 2)
yy = model.predict(xx).reshape(*xx1.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 15))
ax.scatter(X[:, 0], X[:, 1], y, s=80, color='grey')
surf = ax.plot_surface(
    xx1,
    xx2,
    yy,
    linewidth=0, 
    antialiased=False,
    alpha=0.4
)
plt.show()
del fig, ax, surf


print(
    f'{model.dof} -- degrees of freedom\n'
    f'{model.sigma_sqrd} -- sigma squared\n'
    f'{model.beta} -- optimal beta\n'
    f'{model.var_beta}\n -- variance of beta\n'
)

interval_type = 'pred'
sigma_sqrd = model.sigma_sqrd

var_f, pred_conf_interval = pci.glm_pred_conf_interval(
    model,
    X, 
    y, 
    sample_weights=None,
    x_test=xx,
    interval_type=interval_type, 
    significance_level=None
)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 15))
ax.scatter(X[:, 0], X[:, 1], y, s=80, color='grey')
surf = ax.plot_surface(
    xx1,
    xx2,
    yy,
    linewidth=0, 
    antialiased=False,
    alpha=0.4
)
surf = ax.plot_surface(
    xx1,
    xx2,
    yy - pred_conf_interval.reshape(yy.shape),
    linewidth=0, 
    antialiased=False,
    alpha=0.15,
    color='purple'
)
surf = ax.plot_surface(
    xx1,
    xx2,
    yy + pred_conf_interval.reshape(yy.shape),
    linewidth=0, 
    antialiased=False,
    alpha=0.15,
    color='purple'
)
plt.show()
del fig, ax, surf
