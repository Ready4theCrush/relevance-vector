# Relevance-Vector

This library is a relevance vector machine regression implementation using the method set out in "Fast Marginal Likelihood Maximisation for Sparse Bayesian Models" by Michael E. Tipping and Anita Faul at while at Microsoft Research in 2003. (**mostly**, convergence works a little different in this package)

##Usage

###Setup

```javascript
var RVM = require('relevance-vector');
var rvm = new RVM(params);
```

Where 
```javascript
var params = {
    kernel: {
        type: "rbf",
        sigma: [sigma in the radial basis kernel function],
        normalize: [DEFAULT: true],
        bias: [DEFAULT: true]
    },
    min_L_factor: [DEFAULT: 1],
    verbose: [DEFAULT: false]
}
```

###Options
* `kernel.type`: This has to be "rbf" (radial basis function), because that's the only kernel implemented so far.
* `kernel.sigma`: The radial basis kernel function is exp( - ||basis1 - basis2||^2 / 2*sigma^).
* `kernel.normalize`: True normalizes basis vectors before they are kernelized.
* `kernel.bias`: True adds a constant vector of all 1's to the kernel matrix.
* `min_L_factor: As the model builds, it evaluates the likelihood that adding/deleting/modifying a basis vector in/not in the model will improve the model. This factor specifies how small those likelihoods need to be before the model is converged. A large value (10) would mean the model would converge quickly, and not be very specific. A small value (.001) means the model will keep going until it really doesn't think changing anything will help very much. Because max Likelihood scales ~ N (the number of training vectors), min_L = min_L_factor*N.
* `verbose`: Shows the model evaluating the modification that yields the maximum likelihood of improvement. Also shows the min_L it needs to be below to converge.

###Train

```javascript
rvm.train({
    design: [array of N arrays of feature vectors],
    target: [array of N target values]
});
```

*`design`: A 2d array of feature vectors. If there are N feature vectors of length M, this array is N by M (so `design.length` = N, `design[0].length` = M).
*`target`: A 1d array of target values. (`target[i]` = the ith target value)

###Predict

```javascript
rvm.predict({
    design: [array of P arrays of feature vectors],
    target (OPTIONAL): [array of P target values you can include if you want to check them against the predictions for validation]
```

Output:

```javascript
{
    predicted: [array of P predicted values from the provided feature vectors],
    variance: [variance for each prediction. This model assumes a Gaussian distribution for the predicted values],
    kernel_avg: [an average of the off-diagonal values in the kernel matrix. Useful for diagnosing if sigma is too big or too small],
    target (OPTIONAL): [the target values provided for validation, passed straight through w/out any changes],
    error (OPTIONAL): [the target value - the predicted value for each feature vector],
    rmse (OPTIONAL): [the rmse of the predicted values vs. the validation target values]
}
```

