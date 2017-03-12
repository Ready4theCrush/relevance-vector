# Relevance-Vector

This library is a relevance vector machine regression implementation using the method set out in ["Fast Marginal Likelihood Maximisation for Sparse Bayesian Models" by Michael E. Tipping and Anita Faul](http://www.miketipping.com/papers/met-fastsbl.pdf) at while at Microsoft Research in 2003. (**mostly**, convergence works a little different in this package)

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
        type: "rbf",        // at this point, the only kernel function implemented
        sigma: 0.2,         // sigma in the radial basis kernel function (scalar)
        normalize: false,   // DEFAULT: true 
        bias: false         // DEFAULT: true
    },
    min_L_factor: 0.1,      // DEFAULT: 1
    verbose: true           // DEFAULT: false
}
```

###Options
* `kernel.type`: This has to be "rbf" (radial basis function), because that's the only kernel implemented so far.
* `kernel.sigma`: The radial basis kernel function is `exp( - ||basis1 - basis2||^2 / 2*sigma^2)`.
* `kernel.normalize`: True normalizes basis vectors before they are kernelized.
* `kernel.bias`: True adds a constant vector of all `1`'s to the kernel matrix.
* `min_L_factor`: As the model builds, it evaluates the likelihood that adding/deleting/modifying a basis vector in/not in the model will improve the model. This factor specifies how small those likelihoods need to be before the model is converged. A large value (`10`) would mean the model would converge quickly, and not be very specific. A small value (`.001`) means the model will keep going until it really doesn't think changing anything will help very much.
* `verbose`: Shows the model evaluating the modification that yields the maximum likelihood of improvement. Also shows the `min_L` it needs to be below to converge.

###Train

```javascript
rvm.train({
    design: [..], //array of N arrays of feature vectors
    target: [..]  //array of N target values
});
```

* `design`: A 2d array of feature vectors. If there are N feature vectors of length M, this array is N by M (so `design.length` = N, `design[0].length` = M).
* `target`: A 1d array of target values. (`target[i]` = the ith target value)

###Predict

```javascript
var output = rvm.predict({
    design: [..],           //array of P arrays of feature vectors
    target: [OPTIONAL..]    //array of P target values included for validation
});
```

Where `output` has properties:

* `predicted`: An array of P predicted values from the provided feature vectors.
* `variance`: Variance for each prediction. This model assumes a Gaussian distribution for each of the predicted values.
* `kernel_avg`: Mean of the off-diagonal values in the kernel matrix. Useful for diagnosing if sigma is too big or too small.

**provided only if `target` provided to `rvm.predict()`**

* `target`: The target values provided for validation without change.
* `error`: `target - predicted`.
* `rmse`: The root mean squared error of the predicted values against the validation target values.

##How this works
The relevance vector machine is a sparse Bayesian supervised learning algorithm. "Supervised learning" means that you give the rvm a bunch of example data to train it, then use the model you created to make predictions of new data. "Sparse" means that the model created doesn't include very many vectors. Prediction is very fast. "Bayesian" means this is a probabilistic model - if you give the RVM some data to use for predictions it doesn't respond with just an answer, it responds with a mean and variance of a Gaussian probability distribution. It tells you an answer, and how certain it is.

##Values to Use

The RVM kernelizes the data - which means it turns all of the training data points into examples, and then decides which examples it needs to keep to explain the data.

###Sigma

 `Sigma` determines how picky the model is about how similar two data points need to be in order to be considered related.
 
 If sigma is too big, the model will think all the data points are similar to all the other ones, and that there's really just one general kind of example with target values scattered around randomly. Training will be quick because very few vectors are needed to explain the data. Every prediction will be basically an average of the target data used for training.
 
 If sigma is too small, the model will think each example is it's own island - the model will decide many of the vectors are "relevant" to explain the data, and it will take forever to run. Prediction will be so nuanced, the model may decide the feature vector is too far from any of the basis vectors and predict ~0 as the mean and variance.
 
 `kernel_avg` is provided to help find a good `sigma`. `kernel_avg > 0.99` may mean sigma is too big and the model thinks all the examples are too similar. `kernel-avg < 0.1` may mean your prediction feature vectors won't be close to any of the relevance vectors chosen by the model.
 
 You can try `0.2`, `0.5`, `1` and see what happens.

###min_L_factor

`min_L_factor` determines how closely the model needs to be able to explain the training data.

At each step the model looks at each basis vector included or not included in the model and decides if adding/deleting or changing the level of confidence in the basis vector will improve the model. A likelihood of improvement is assigned to the change. Whatever change has the maximum likelihood is the one implemented at that step. Why `min_L_factor` instead of `min_L`? `min_L` is the minimum likelihood required for a change to be "worth it" and for the model to keep going instead of considering itself converged. `min_L` seems to related to N, and it's a pain to have to change it every time N changes. `min_L = N*min_L_factor` for the convergence check at each step.

A large `min_L_factor` means the model will train quickly, but won't try too hard to be accurate. This great for getting a rough estimate.

A small `min_L_factor` means the model will iterate many times trying to eek out the tiniest improvements in likelihood you will be able to explain the target data. Great for a more accurate model...to a point, at some point more precision doesn't help the model against validation data.

You can try `1`, `0.1`, `.01` to start.

##Caveat

If the feature vectors used for predictions don't project onto the relevance vectors included in the model, the model will predict a value of near zero for the mean and variance (if `bias: false`). This could be because sigma was too small, and the model doesn't think anything is similar to anything else. This could be because the new feature vector(s) is just really different from anything in the training set. If you see a `3.222e-20` in your predictions, it's probably not because that's what makes sense. If `bias: true`, it's harder to pick out if the prediction is just `bias + 0` or an actual good prediction.

##Patent

Microsoft owns a patent on an implementation of the Relevance Vector Machine. But not this implementation. They hold a patent on the [first algorithm Tipping created](http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf). The one used in this package is based on [this paper](http://www.miketipping.com/papers/met-fastsbl.pdf). At least that's what is seems like from looking at [the patent](https://www.google.com/patents/US6633857).

If anyone from the Microsoft legal team disagrees, feel free to let me know :-)