var m = require('mathjs');

var h = {};

h.get_max_and_index = function(arr) {
    var max = arr[0];
    var maxIndex = 0;
    
    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    
    return maxIndex;
};

h.get_phi_and_A = function(p) {
    p.bases = [];
    // console.log("p.alphas: ");
    // console.log(p.alphas);
    p.used_alphas = p.alphas._data[0].filter(function(alpha, index) {
        if (alpha < p.INF) {
            p.bases.push(index);
            return true;
        } else {
            return false;
        }
    });
    
    if (p.bases.length == 0) {
        var restart_basis = m.randomInt(0, p.M);
        p.alphas._data[0][restart_basis] = 0.1;
        p.bases = [restart_basis];
        p.used_alphas = [0.1];
    }
    
    // cl("used_alphas: ");
    // cl(p.used_alphas);
    
    // steps:
        // 1.) for each base, get the basis vector
        // 2.) make each vector horizontal and grab the array of the vector
        // 3.) make that into a matrix
        // 4.) transpose that matrix so the bases are vertical again. 
    p.phi = p.kernel_mtx.subset(m.index(
        m.range(0, p.N),
        p.bases
    ));
    // cl("p.phi: ");
    // cl(p.phi);
    // p.phi = m.transpose(
        // m.matrix(
            // p.bases.map(function(base) {
                // var basis_as_row = m.transpose(
                    // p.basis_by_index[base]
                // )._data;
                // return basis_as_row;
            // })
        // )
    // );
    p.A = m.diag(p.used_alphas);
    // cl("p.A: ");
    // cl(p.A);
    return p;
}
        
h.get_pred_var = function(p) {
    var pred_var = p.phi_proj._data.map(function(phi_x_array) {
        var phi_x = m.matrix([phi_x_array]);
        var x_variance = m.add(
            m.inv(p.beta),
            m.multiply(
                m.multiply(
                    phi_x,
                    p.cov
                ),
                m.transpose(phi_x)
            )._data[0][0]
        )
        return x_variance;
    });
    return pred_var;
};
    
h.get_cov = function(p) {
    var cov = m.inv(
        m.add(
            p.A,
            m.multiply(
                p.beta,
                m.multiply(
                    m.transpose(p.phi),
                    p.phi
                )
            )
        )
    );
    return cov;
};

h.get_mu = function(p) {
    var mu = m.multiply(
        p.beta,
        m.multiply(
            p.cov,
            m.multiply(
                m.transpose(p.phi),
                p.t
            )
        )
    );
    // if (p.bases.length == 2 && p.mu._size[0] == 1) {
        // console.log("WTF: ");
        // console.log(mu);
    // }
    // cl("mu: ");
    // cl(mu);
    return mu;
};

h.get_Q_precomp = function(p) {

    // cl("p.cov: ");
    // cl(p.cov);
    // cl("p.beta: ");
    // cl(p.beta);

    var beta_phi_cov =  m.multiply(
        p.beta,
        m.multiply(
            p.phi,
            p.cov
        )
    );
    
    var phit_beta_target = m.multiply(
        m.multiply(
            m.transpose(p.phi),
            p.beta
        ),
        p.t
    );
    
    var beta_phi_cov_phit_beta_target = m.multiply(
        beta_phi_cov,
        phit_beta_target
    );

    return beta_phi_cov_phit_beta_target;
}

h.get_Q_precomp_classifier = function(p) {
    // console.log("p.beta size: "+p.beta._size);
    // console.log("p.cov size: "+p.cov._size);
    var beta_phi_cov_phit_beta = m.multiply(
        p.beta,
        m.multiply(
            p.phi,
            m.multiply(
                p.cov,
                m.multiply(
                    m.transpose(p.phi),
                    p.beta
                )
            )
        )
    );
    // cl("beta_phi_cov_phit_beta");
    // cl(beta_phi_cov_phit_beta);
    
    return beta_phi_cov_phit_beta;
}
    

h.get_Q = function(p) {
    // cl("p.basis_beta_t: ");
    // cl(p.basis_beta_t);
    // cl("trans p.basis: ");
    // cl(m.transpose(p.basis));
    // cl("p.Q_precomp: ");
    // cl(p.Q_precomp);
    var Q = m.subtract(
        p.basis_beta_t,
        m.multiply(
            m.transpose(p.basis),
            p.Q_precomp
        )
    );
    // cl("Q._data: ");
    // cl(Q._data);
    return Q._data[0][0];
};

h.get_S_classifier = function(p) {
    var first_term = m.multiply(
        m.transpose(p.basis),
        m.multiply(
            p.beta,
            p.basis
        )
    );
    
    var second_term = m.multiply(
        m.transpose(p.basis),
        m.multiply(
            p.beta_phi_cov_phit_beta,
            p.basis
        )
    );
    
    var S = m.subtract(first_term, second_term)._data[0][0];
    cl("S: "+S);
    return S;
}

h.get_Q_classifier = function(p) {
    var first_term = m.multiply(
        m.transpose(p.basis),
        m.multiply(
            p.beta,
            p.t_hat
        )
    );
    
    // cl("p.t_hat: ");
    // cl(p.t_hat);
    // cl(p.beta_phi_cov_phit_beta);
    var second_term = m.multiply(
        m.transpose(p.basis),
        m.multiply(
            p.beta_phi_cov_phit_beta,
            p.t_hat
        )
    );
    
    var Q = m.subtract(first_term, second_term)._data[0][0];
    cl("Q: "+Q);
    return Q;
};

h.get_S = function(p) {
    var basist_phi_beta = m.multiply(
        m.multiply(
            m.transpose(p.basis),
            p.phi
        ),
        p.beta
    );
    
    var second_term = m.multiply(
        m.multiply(
            basist_phi_beta,
            p.cov
        ),
        m.transpose(basist_phi_beta)
    );
    
    var diff = m.subtract(
        p.basisnorm_beta,
        second_term
    );
    
    // cl("S: ");
    // cl(diff);
    
    if (typeof diff == 'object') {
        return diff._data[0][0];
    } else {
        return diff;
    }
}

h.get_s = function(p) {
    if (p.a < p.INF) {
        var s = m.divide(
            m.multiply(
                p.a,
                p.S
            ),
            m.subtract(
                p.a,
                p.S
            )
        );
    } else {
        var s = p.S
    }
    // cl("s: ");
    // cl(s);
    return s;
}

h.get_q = function(p) {
    if (p.a < p.INF) {
        var q = m.divide(
            m.multiply(
                p.a,
                p.Q
            ),
            m.subtract(
                p.a,
                p.S
            )
        );
    } else {
        var q = p.Q
    }
    // cl("q: ");
    // cl(q);
    return q;
}

h.get_a_new = function(p) {
    var theta = m.subtract(
        m.pow(p.q, 2),
        p.s
    );
    
    if (theta > 0) {
        var a = m.divide(
            m.pow(p.s, 2),
            theta
        );
    } else {
        var a = p.INF
    }
    return a;
}

h.get_delta_L = function(p) {
    if (p.a == p.INF && p.a_new < p.INF) {
        var delta_L = m.add(
            m.divide(
                m.subtract(
                    m.pow(p.Q, 2),
                    p.S
                ),
                p.S
            ),
            m.log(
                m.divide(
                    p.S,
                    m.pow(p.Q, 2)
                )
            )
        );
    } else if (p.a < p.INF && p.a_new < p.INF) {
        var delta_L = m.subtract(
            m.divide(
                m.pow(p.Q, 2),
                m.add(
                    p.S,
                    m.inv(
                        m.subtract(
                            m.inv(p.a_new),
                            m.inv(p.a)
                        )
                    )
                )
            ),
            m.log(
                m.add(
                    1,
                    m.multiply(
                        p.S,
                        m.subtract(
                            m.inv(p.a_new),
                            m.inv(p.a)
                        )
                    )
                )
            )
        );
    } else if (p.a < p.INF && p.a_new == p.INF) {
        var delta_L = m.subtract(
            m.divide(
                m.pow(p.Q, 2),
                m.subtract(
                    p.S,
                    p.a
                )
            ),
            m.log(
                m.subtract(
                    1,
                    m.divide(
                        p.S,
                        p.a
                    )
                )
            )
        );
    } else {
        var delta_L = -1.0*p.INF;
    }
    return delta_L;
}

h.get_beta = function(p) {
    var num_used = p.cov._size[0];
    // try {
    // m.subtract(
        // p.t,
        // m.multiply(
            // p.phi,
            // p.mu
        // )
    // );
    // } catch (e) {
        // console.log(e);
        // cl("p.t._size: "+p.t._size);
        // cl("p.phi._size: "+p.phi._size);
        // cl("p.mu._size: "+p.mu._size);
    // }
    
    var est_err = m.subtract(
        p.t,
        m.multiply(
            p.phi,
            p.mu
        )
    );
    
    // cl("est_err: ");
    // cl(est_err);
    
    var est_norm = m.multiply(
        m.transpose(est_err),
        est_err
    );
    
    // cl('est_norm: ');
    // cl(est_norm);
    
    var NM_diff = m.subtract(
        p.N,
        num_used
    );
    
    // cl("m.diag(p.used_alphas): ");
    // cl(m.diag(p.used_alphas));
    
    // cl("m.diag(m.diag(p.cov)): ");
    // cl(m.diag(m.diag(p.cov)));
    var a_cov_diag_trace = m.sum(
        m.diag(
            m.multiply(
                m.diag(p.used_alphas),
                m.diag(m.diag(p.cov))
            )
        )
    );
    
    // cl("a_cov_diag_trace: ");
    // cl(a_cov_diag_trace);
    
    var beta = m.inv(
        m.divide(
            est_norm,
            m.add(
                NM_diff,
                a_cov_diag_trace
            )
        )
    );
    
    // cl("beta: ");
    // cl(beta._data);
    
    return beta._data[0][0];
};

h.sigmoid = function(y) {
    return m.divide(
        1,
        m.add(
            1,
            m.exp(-1*y)
        )
    );
};

h.get_y = function(p) {
    var phi_mu = m.multiply(
        p.phi,
        p.mu
    );
    return phi_mu.map(h.sigmoid);
};

h.get_pred_y_classifier = function(p) {
    // cl("p.phi_proj: ");
    // cl(p.phi_proj);
    var phi_mu = m.multiply(
        p.phi_proj,
        p.mu
    );
    return phi_mu.map(h.sigmoid);
}
                
h.get_beta_classifier = function(p) {
    // console.log("p.y: ");
    // console.log(p.y._data);
    var b_diag = p.y.map(function(yn) {
        return m.divide(
            yn,
            m.subtract(
                1,
                yn
            )
        );
    });
    b_diag = m.transpose(b_diag)._data[0];
    // console.log("b_diag: ");
    // console.log(b_diag);
    // console.log("m.diag(b_diag)");
    // console.log(m.diag(b_diag));
    // return m.diag(m.transpose(b_diag)._data[0]);
    return m.matrix(m.diag(b_diag));
}

h.get_cov_classifier = function(p) {
    // console.log("p.phi size: "+p.phi._size);
    // cl("p.phi: ");
    // cl(p.phi);
    // cl("p.beta: ");
    // cl(p.beta);
    var cov = m.inv(
        m.add(
            m.multiply(
                m.multiply(
                    m.transpose(p.phi),
                    p.beta
                ),
                p.phi
            ),
            p.A
        )
    );
    // cl("cov: ");
    // cl(cov._data);
    return cov;
}

h.get_t_hat_classifier = function(p) {
    // console.log("p.y size: "+p.y._size);
    // console.log("p.t size: "+p.t_size);
    var t_hat = m.add(
        m.multiply(
            p.phi,
            p.mu
        ),
        m.multiply(
            m.inv(p.beta),
            m.subtract(
                p.t,
                p.y
            )
        )
    );
    cl("t_hat: ");
    cl(t_hat);
    return t_hat;
};

h.get_prediction_classifier = function(p) {
    return m.add(
        m.multiply(
            p.proj_phi,
            p.mu
        ),
        m.multiply(
            m.inv(p.beta),
            m.subtract(
                p.t,
                p.pred_y
            )
        )
    );
}

h.get_mu_classifier = function(p) {
    // console.log("p.beta size: "+p.beta._size);
    // console.log("p.t_hat size: "+p.t_hat._size);
    // var mu = m.multiply(
        // m.multiply(
            // p.cov,
            // m.transpose(p.phi)
        // ),
        // m.multiply(
            // p.beta,
            // p.t_hat
        // )
    // );
    var diff = m.subtract(p.t, p.y);
    var mu = m.multiply(
        m.inv(p.A),
        m.multiply(
            m.transpose(p.phi),
            diff
        )
    );
    return mu;
};

h.initialize = function(p){
    p.alphas = m.multiply(
        m.ones(1, p.M),
        p.INF
    );
    
    //sets beta = 10 / var(t)
    p.beta = m.divide(
        10,
        m.var(p.t)
    );

    // console.log("p.beta: ");
    // console.log(p.beta);
    // cl("p.alphas: ");
    // cl(p.alphas);
    
    p.bases = [];
        
    //Find the initial basis by finding the maximum projection of the basis onto the target vector
    
    var L = p.alphas._data[0].map(function (val, index) {
        // cl(p.basis_t_by_index[index]);
        // cl(p.basis_inner_product[index]);
        // cl(m.divide(p.basis_t_by_index[index], p.basis_inner_product[index]));
        return  p.basis_t_by_index[index] / p.basis_inner_product[index];
    });
    
    // console.log("L: ");
    // console.log(L);
    
    var maxLIndex = h.get_max_and_index(L);
    
    //get the initial value for this first alpha
    p.alphas._data[0][maxLIndex] = m.divide(
        p.basis_inner_product[maxLIndex],
        m.subtract(
            m.divide(
                p.basis_t_by_index[maxLIndex],
                p.basis_inner_product[maxLIndex]
            ),
            m.divide(
                1,
                p.beta
            )
        )
    );
    // console.log("p.alphas: ");
    // console.log(p.alphas);    
    
    p = h.get_phi_and_A(p);
    p.cov = h.get_cov(p);
    p.mu = h.get_mu(p);
    

    
    if (p.method == "classification") {
        p.beta = m.matrix([[p.beta]]);
        // cl("in initialize");
        // cl("p.beta: ");
        // cl(p.beta);
        // p.y = h.get_y(p);
        p.t_hat = m.multiply(m.ones(p.N, 1), 0.5);
    }
    
    return p;
}

h.normalize_design = function(design) {
    var design_transpose = m.transpose(design); //to make each basis it's own entry in an array
    var basis_lengths = design_transpose.map(function(basis) {
        var squared_basis = m.map(basis, function (element) {
            return m.pow(element, 2);
        });
        var length = m.sqrt(m.sum(squared_basis));
        if (length == 0) {
            length = 1;
        };
        return length;
    });
    var normalized_transpose = design_transpose.map(function (basis, index) {
        return m.divide(basis, basis_lengths[index]);
    });
    var normalized = m.transpose(normalized_transpose);
    return {
        normalized: normalized,
        pre_k_basis_lengths: basis_lengths
    }
}

h.normalize_features = function(pre_k_basis_lengths, design) {
    var design_transpose = m.transpose(design);
    var normalized_transpose = design_transpose.map(function(basis, index) {
        return m.divide(basis, pre_k_basis_lengths[index]);
    });
    return m.transpose(normalized_transpose);
}

function cl(obj) {
    console.log(obj);
}

module.exports = h;

