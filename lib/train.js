var m = require('mathjs');
var k = require('./kernel.js');
var h = require('./helpers.js');

function train_add(data) {
    if (this.design[0].length != data.design[0].length) {
        throw 'new design features are different length from the old ones';
    }
    try {
        if (this.kernel.normalize) {
            data.design = h.normalize_design(data.design);
        };
        this.design.concat(data.design);
        var target_vec = m.transpose(this.target)._data[0];
        target_vec.concat(data.target);
        this.target = m.transpose(m.matrix(target_vec));
    } catch(e) {
        console.log("there was a problem adding to the design and target arrays");
        throw e;
    }
    
    // check if dimensions for training are ok
    if (data.design.length != data.target.length) {
        var designTargetLenghtMismatch = `design matrix has ${data.design.length} vectors and target vector has ${data.target.length} values. They're supposed to be the same.`;
        throw designTargetLenghtMismatch;
    }
    
    //R is the size of the data set before adding more.
    this.R = this.kernel_mtx._size[0];
    this.kernel_mtx = k.append_kernel_mtx({
        kmtx_orig: this.kernel_mtx,
        kernel: this.kernel,
        design: this.design
    });
    
    this.method = data.method || "regression";
    this.M = this.kernel_mtx._size[1];
    this.N = this.design.length;
    
    // console.log(this.kernel_mtx);    
    
    var kernel_transpose = m.transpose(this.kernel_mtx);
    //the goal here is to add the extra vectors to this precomp object
    this.basis_by_index = this.basis_by_index.concat(
        kernel_transpose._data.slice(R).map(function (col) {
            return m.transpose(m.matrix(col));
        })
    );
}

    

function train(data) {
    try {
        var adjusted = h.normalize_design(data.design);
        this.design = adjusted.normalized;
        this.pre_k_basis_lengths = adjusted.pre_k_basis_lengths;
        var tar = m.zeros(data.target.length, 1);
        data.target.forEach(function(element, index) {
            tar._data[index][0] = element;
        });
        this.target = tar;  //target is a vertical vector
        // console.log("this.target: ");
        // console.log(this.target);
    } catch (e) {
        console.log("there was a problem turning the design and target arrays into matrices");
        throw e;
    }
    
    // check if dimensions for training are ok
    if (data.design.length != data.target.length) {
        var designTargetLenghtMismatch = `design matrix has ${data.design.length} vectors and target vector has ${data.target.length} values. They're supposed to be the same.`;
        throw designTargetLenghtMismatch;
    }

    this.kernel_mtx = k.get_kernel_mtx({
        design: this.design,
        kernel: this.kernel}
    );
    
    // cl("this.kernel_mtx: ");
    // cl(this.kernel_mtx._data);

    console.log("method: "+this.method);
    this.M = this.kernel_mtx._size[1];
    this.N = this.design.length;
    
    var proj_sum = m.sum(m.subset(
        this.kernel_mtx,
        m.index(
            m.range(0, this.N),
            m.range(+0+this.kernel.bias, this.M)
        )
    ));
    var proj_count = m.subtract(m.pow(this.N, 2), this.N);
    this.kernel_avg = m.divide(proj_sum, proj_count);
    
    //make some objects that will help with preprocessing of stuff
    
    // if (this.method == "regression") {
        var precomps = regression_precomps(this.kernel_mtx, this.target);
        this.basis_by_index = precomps.basis_by_index;
        this.basis_inner_product = precomps.basis_inner_product;
        this.basis_t_by_index = precomps.basis_t_by_index
    // }
    
    this.INF = 1000000000; //a big number used to check big numberness
    
    //this is useful for holding stuff needed during the iterations.
    var params = {
        M: this.M,
        N: this.N,
        method: this.method,
        bias: this.kernel.bias,
        bases: [],
        alphas: {},
        used_alphas: [],
        A: {},
        phi: {},
        cov: {},
        mu: {},
        t: this.target,
        min_L: this.min_L_factor*this.N,
        kernel_mtx: this.kernel_mtx,
        kernel_transpose: this.kernel_transpose,
        basis_by_index: this.basis_by_index,
        basis_inner_product: this.basis_inner_product,
        basis_t_by_index: this.basis_t_by_index,
        INF: this.INF,
        verbose: this.verbose
    };

    params = h.initialize(params);
    
    params = get_model(params);
    // console.log("params.bases: "+params.bases);
    this.cov = params.cov;
    this.mu = params.mu;
    this.alphas = params.alphas;
    this.bases = params.bases;
    this.beta = params.beta;
    this.t = params.t;
    console.log("this.bases: "+this.bases);

}    
    
function get_model(params) {
    
    var converged = false;
    
    //start the loop that finds the cov, mean    
    for (var r = 0; !converged; r++) {
        if (params.method == "regression") {
            params.beta = h.get_beta(params);
            params = h.get_phi_and_A(params);
            params.cov = h.get_cov(params);
            params.mu = h.get_mu(params);
            params.Q_precomp = h.get_Q_precomp(params);
        } else {
            params.y = h.get_y(params);
            params.beta = h.get_beta_classifier(params);    
            params = h.get_phi_and_A(params);
            params.cov = h.get_cov_classifier(params);
            params.mu = h.get_mu_classifier(params);
            params.beta_phi_cov_phit_beta = h.get_Q_precomp_classifier(params);
            params.t_hat = h.get_t_hat_classifier(params);
        }
        params.max_L = -params.INF;
        params.max_L_index = 0;
        params.max_L_a_new = 0;
        
        for (var ind = 0; ind < params.alphas._data[0].length; ind++) {
            params.a = params.alphas._data[0][ind];
            params.basis = params.basis_by_index[ind];
            if (params.method === "regression") {
                params.basisnorm_beta = m.multiply(
                    params.basis_inner_product[ind],
                    params.beta
                );
                params.basis_beta_t = m.multiply(
                    params.basis_t_by_index[ind],
                    params.beta
                );
                params.Q = h.get_Q(params);
                params.S = h.get_S(params);
            } else {
                params.Q = h.get_Q_classifier(params);
                params.S = h.get_S_classifier(params);
            }
            params.q = h.get_q(params);
            params.s = h.get_s(params);
            params.a_new = h.get_a_new(params);
            params.delta_L = h.get_delta_L(params);
            if (params.delta_L > params.max_L) {
                params.max_L = params.delta_L;
                params.max_L_index = ind;
                params.max_L_a_new = params.a_new;
            }
            // console.log("Q: "+params.Q+", S: "+params.S+", delta_L: "+params.delta_L);
        }
        
        if (params.verbose) {console.log("min_L: "+params.min_L+", max_L: "+params.max_L)};
        // console.log(params.alphas._data);
        converged = params.max_L < params.min_L && r > 3
        // converged = m.log(m.abs(params.alphas._data[0][params.max_L_index] - params.max_L_a_new)) < -6.0;
        if (!converged) {
            // console.log("delta_a: "+m.log(m.abs(params.alphas._data[0][params.max_L_index] - params.max_L_a_new)));
            params.alphas._data[0][params.max_L_index] = params.max_L_a_new;
        }
    }
    // console.log("params.bases: "+params.bases);
    return params;
}

function regression_precomps(kernel, target) {

    var kernel_transpose = m.transpose(kernel);
    // cl("kernel: ");
    // cl(kernel);

    //given an index, this array gives the vertical vector (as mathjs matrix) of that basis vector.
    var basis_by_index = kernel_transpose._data.map(function (col) {
        return m.transpose(m.matrix([col]));
    });
    
    // cl("basis_by_index");
    // cl(basis_by_index.map(function(basis) {return basis._data;}));

    //given an index, this has the inner product of the basis at that index, with itself.    
    var basis_inner_product = basis_by_index.map(function (basis) {
        return m.multiply(
            m.transpose(basis),
            basis
        )._data[0][0]; //this is a scalar, getting [0][0] gets us the scalar part.
    });
    
    // cl("basis_inner_product: ");
    // cl(basis_inner_product);
    
    //given an index, this gives the inner product of the basis at that index, and the vector of target values
    // cl("basis_by_index[0]._size: "+basis_by_index[0]._size);
    // cl("target._size: "+target._size);
    // cl(basis_by_index.map(function(basis) {return basis._data;}));
    var basis_t_by_index = basis_by_index.map(function (basis) {
        return m.multiply(
            m.transpose(basis),
            target
        )._data[0][0];
    });
    
    // cl(basis_t_by_index.map(function(bt) {return bt;}));
    
    return {
        basis_by_index: basis_by_index,
        basis_inner_product: basis_inner_product,
        basis_t_by_index: basis_t_by_index
    };
}

function cl(obj) {
    console.log(obj);
}
    
module.exports = train;

