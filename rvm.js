var CustomError = require('./lib/custom_error.js');
var train = require('./lib/train.js');
var predict = require('./lib/predict.js');

function RVR(params) {
        if (params.kernel && params.kernel.type == "rbf" && params.kernel.sigma) {
            this.kernel = params.kernel;
            // console.log(params);
            if (!params.kernel.hasOwnProperty("bias")) {
                // console.log(params.kernel.hasOwnProperty("bias"));
                this.kernel.bias = true;
            }
            if (!params.kernel.hasOwnProperty("normalize")) {
                this.kernel.normalize = true;
            }
            this.method = params.method || "regression";
            this.verbose = params.verbose || false;
            this.min_L_factor = params.min_L_factor || 1;
        } else {
            throw "params.kernel needs to have .type = rbf and .kernel = sigma"
        }
        
        this.train = train;
        this.predict = predict;
}

RVR.prototype.describe = function() {
    console.log({
        kernel: this.kernel,
        verbose: this.verbose,
        min_L_factor: this.min_L_factor,
        method: this.method
    });
}

module.exports = RVR;
    