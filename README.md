# General Linear Model

Simple repository for finding solutions <img src="https://latex.codecogs.com/svg.image?\widetilde{y}&space;=&space;f(x,&space;\widetilde{\beta})" title="\widetilde{y} = f(x, \widetilde{\beta})" />, to models of the form: <img src="https://latex.codecogs.com/svg.image?f(x,&space;\vec{\beta})&space;=&space;A(x)&space;\cdot&space;&space;\vec{\beta}" title="f(x, \vec{\beta}) = A(x) \cdot \vec{\beta}" /> .  Where A is an mxn matrix, whose columns are the basis functions comprising the model, and the rows are the basis functions evaluated at the sample points.

When x is also a vector (in a multi-dimensional model), A(x) simply gains new columns (grouped by each new coordinate) to accomodate higher dimensional models.

Also included are methods to produce prediction and confidence intervals, via both analytic and bootstrapped methods.

## Running the GLM regressor class

### Instanciate GLM class

Parameters:  

basis_funcs: tuple of lists   
    Basis functions provided as a list or lists or tuple of lists, where each list of basis functions is for the appropriate dimension in the problem.  If 1D then only a single list, within the tuple, is needed.  The default is ([lambda x: np.ones_like(x), lambda x: x], ).  
```
import numpy as np
import glm.general_linear_model as glm

model = glm.GLM(basis_funcs=([lambda x: np.ones_like(x), lambda x: x], ))
```

### Fit
Parameters:  

X: array like of shape (n_samples, n_features)  
    Feature variables with columns as independent dimensions.  e.g. if inputs are 3-dimensional, one column each for x, y, z.  

y: array like of shape (n_samples, )  
    Vector of targets.  

sample_weights -- array like of shape (n_samples, ), optional  
    Sample weights for the target.  The default is None.  

Returns:  

None
```
model.fit(X, y, sample_weights=None)
```

### Predict
Parameters:  

X: array like of shape (n_samples, n_features)  
    Feature variables with columns as independent dimensions.  e.g. if inputs are 3-dimensional, one column each for x, y, z.  

Returns:  

y: array like of shape (n_samples, )  
    Predicted values.  
```
y_fit = model.predict(X)
```

## Making Prediction or Confidence Intervals
The two methods available with both return a variance of the function (var_f) and the prediction or condifence interval (pred_conf_interval).  The interval provided is intended to be used like <img src="https://latex.codecogs.com/svg.image?\widetilde{y}&space;\pm&space;" title="\widetilde{y} \pm " /> interval.

### Analytic

Parameters:  

model: GLM model object  
    Instance of the fitted GLM class.

X: array like of shape (n_samples, n_features)  
    Feature variables with columns as independent dimensions.  e.g. if inputs are 3-dimensional, one column each for x, y, z.  

y: array like of shape (n_samples, )  
    Vector of targets.  

sample_weights -- array like of shape (n_samples, ), optional  
    Sample weights for the target.  The default is None.  
    
x_test: array like of shape (n, )  
    Use to provide a new array to evaluate the intervals over.  The default is None.  

interval_type: str  
    String used to selected between prediction and confidence intervals.  The default is 'pred'.  
    
significance_level: float  
    Value between 0 and 1 to select the significance level used in the percent point function for the students_t distribution.  The default is None.

Returns:  

var_f: array like of shape (n, )  
    Variance of the fitted function of the domain of x_test.  
    
pred_conf_interval: array like of shape (n, )  
    Prediction or confidence interval, on the domain of x_test, based on the interval_type.  

```
import glm.prediction_confidence_interval as pci

var_f, pred_conf_interval = pci.glm_pred_conf_interval(
  model,
  X,
  y,
  sample_weights=None,
  x_test=None,
  interval_type='pred', 
  significance_level=None
)
```
### Bootstrapped

Parameters:  

model: GLM model object  
    Instance of the fitted GLM class.

X: array like of shape (n_samples, n_features)  
    Feature variables with columns as independent dimensions.  e.g. if inputs are 3-dimensional, one column each for x, y, z.  

y: array like of shape (n_samples, )  
    Vector of targets.  

sample_weights -- array like of shape (n_samples, ), optional  
    Sample weights for the target.  The default is None.  
    
x_test: array like of shape (n, )  
    Use to provide a new array to evaluate the intervals over.  The default is None, and will use x_test=x.  

interval_type: str  
    String used to selected between prediction and confidence intervals.  The default is 'pred'.  
    
significance_level: float  
    Value between 0 and 1 to select the significance level used in the percent point function for the students_t distribution.  The default is None.

nbootstraps: int  
    Sets number of bootstrapping iterations.  The default is None, and will use nbootstraps=len(x).  

resample_frac: float  
    Random sampling fraction of the sample set to use for each bootstrap step.  

Returns:  

var_f: array like of shape (n, )  
    Variance of the fitted function of the domain of x_test.  
    
pred_conf_interval: array like of shape (n, )  
    Prediction or confidence interval, on the domain of x_test, based on the interval_type.  

```
import glm.bootstrapped_pred_conf_interval as bpci

var_f, pred_conf_interval = bpci.bootstrap_pred_conf_interval(
  model,
  X,
  y,
  sample_weights=None,
  x_test=None,
  interval_type='pred', 
  significance_level=None,
  nbootstraps=None, 
  resample_frac=0.7
)
```
