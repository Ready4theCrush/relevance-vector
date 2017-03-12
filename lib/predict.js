var m = require('mathjs');
var k = require('./kernel.js');
var h = require('./helpers.js');

module.exports = function predict(data) {
    var new_design = h.normalize_features(this.pre_k_basis_lengths, data.design);
    var feature_length = this.design[0].length;
    var bias_used = this.bases.indexOf(0) > -1 && this.kernel.bias;   
    var shifted_bases = m.subtract(this.bases, +this.kernel.bias);
    var used_bases = shifted_bases.filter(function(shifted) {
        return shifted > -1;
    });
    var used_design = m.subset(
        this.design,
        m.index(
            used_bases,
            m.range(0, feature_length)
        )
    );
    var proj_mtx = k.project_features({
        kernel: this.kernel,
        used_design: used_design,
        new_design: new_design,
        bias_used: bias_used
    });
    // console.log("proj_mtx: ");
    // console.log(proj_mtx);
    // console.log("this.beta: ");
    // console.log(this.beta);
    // console.log("this.mu: ");
    // console.log(this.mu);
    var output = {};
    var params = {
        phi: this.phi,
        mu: this.mu,
        cov: this.cov,
        beta: this.beta,
        t: this.t,
        phi_proj: proj_mtx
    };    
    if (this.method === "regression") {
        var predicted = m.multiply(
            proj_mtx,
            this.mu
        );
        output.variance = h.get_pred_var(params);
        output.predicted = m.transpose(predicted)._data[0];
        output.kernel_avg = this.kernel_avg;
        if (data.target) {
            output.target = data.target;
            output.error = m.subtract(output.target, output.predicted);
            output.rmse = m.sqrt(
                m.mean(
                    output.error.map(function (element) {
                        return m.pow(element, 2);
                    })
                )
            );
        }
    } else {
        // console.log("running classifier case");
        var predicted = h.get_pred_y_classifier(params);
        output.predicted = m.transpose(predicted)._data[0];
        output.variance = m.diag(this.beta);
        // var predicted = h.get_prediction_classifier(params);
    }
    return output;
}