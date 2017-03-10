var m = require('mathjs');

module.exports.get_kernel_mtx = function(p) {
    // console.log("get_phi params: ");
    // console.log(p);
    if (p.kernel.type === 'rbf') {
        var dist_mtx = get_dist_mtx(p.design);
        // console.log("dist_mtx: ");
        // console.log(dist_mtx);
        var kernel = make_rbf_kernel(dist_mtx, p.kernel.sigma);
        // console.log("kernel from make_rbf_kernel");
        // console.log(kernel);
    }
    if (p.kernel.bias) {
        kernel = add_bias(kernel);
    }
    return kernel;
};

module.exports.append_kernel_mtx = function(p) {
    var R = p.kmtx_orig._size[0];
    if (p.kernel.type == "rbf") {
        var dist_mtx = get_dist_mtx(p.design, R);
        var kernel = append_rbf_kernel(
            dist_mtx,
            p.kernel.sigma,
            p.kernel.bias,
            p.kmtx_orig
        );
    }
    return kernel;
}

module.exports.project_features = function(p) {
    var M = p.used_design.length;
    var VN = p.new_design.length;
    var proj_mtx = m.ones(VN, M + p.bias_used);
    if (p.kernel.type == "rbf") {
        var dist_mtx = m.zeros(VN, M);
        // console.log("p.used_design: ");
        // console.log(p.used_design);
        // console.log("p.new_design: ");
        // console.log(p.new_design);
        for (var row = 0; row < VN; row++) {
            for (var col = 0; col < M; col++) {
                var dist = norm_squared(
                    m.matrix(p.new_design[row]),
                    m.matrix(p.used_design[col])
                );
                dist_mtx._data[row][col] = dist;
            }
        };
        // console.log("dist_mtx: ");
        // console.log(dist_mtx);
        // console.log("range: ");
        // console.log(proj_mtx.subset(m.index(
            // m.range(0, VN),
            // m.range(+p.bias_used, M + p.bias_used)
        // )));
        var rbf_dist_mtx = get_rbf_dist(dist_mtx, p.kernel.sigma);
        // console.log("rbf_dist_mtx: ");
        // console.log(rbf_dist_mtx)
        if (M == 1 && VN == 1) {
            rbf_dist_mtx = rbf_dist_mtx._data[0][0];
        }
        proj_mtx.subset(
            m.index(
                m.range(0, VN),
                m.range(+p.bias_used, M + p.bias_used)
            ),
            rbf_dist_mtx
        )
    }
    return proj_mtx;
};

function append_rbf_kernel(dist_mtx, sigma, bias, kmtx_orig) {
    var R = kmtx_orig._size[0];
    var N = dist_mtx._size[0];
    var kernel_add = get_rbf_dist(dist_mtx, sigma);
    // need to do two things:
    // 1.) copy the contents of kernel_add[row, col] to kernel_add[col, row]
    // 2.) copy the kmtx_orig (N by N) to kernel_add so we should have an N_new by N_new mtx with rbf values everywhere.
    for (var row = R; row < N; row++) {
        for (var col = 0; col <= row; col++) {
            kernel_add._data[col, row] = kernel_add._data[row, col];
        }
    };
    m.subset(
        kernel_add,
        m.index(
            m.range(0, R),
            m.range(0, R)
        ),
        kmtx_orig.subset(
            m.range(0, R),
            m.range(0+bias, R+bias)
        )
    );
    if (bias) {
        kernel_add = add_bias(kernel_add);
    }
    return kernel_add;
}

function get_dist_mtx(design, start_row) {
    // so if start_row is null, then we just make a normal N x N matrix.
    // if start_row is not zero, we just make the distances we need to add onto the exisiting distances we have.
    var R = start_row || 0;
    var N = design.length;
    var dist_mtx = m.zeros(N, N);
    for (var row = R; row < N; row++) {
        for (var col = 0; col <= row; col++) {
            var dist = norm_squared(
                m.matrix(design[row]),
                m.matrix(design[col])
            );
            // console.log(`dist at ${row},${col}: ${dist}`);
            dist_mtx._data[row][col] = dist;
        }
    }
    return dist_mtx;
}

function norm_squared(v1, v2) {
    var diff = m.subtract(v1, v2);
    var diff_transpose = m.transpose(diff);
    var norm_squared = m.multiply(diff, diff_transpose);
    return norm_squared;
}

function make_rbf_kernel(dist_mtx, sigma) {
    var kernel = get_rbf_dist(dist_mtx, sigma);
    for (var row = 0; row < dist_mtx._size[0]; row++) {
        for (var col = 0; col <= row; col++) {
            kernel._data[col][row] = kernel._data[row][col];
        }
    }
    // console.log("kernel in make_rbf_kernel");
    // console.log(kernel);    
    return kernel;
}

function add_bias(kernel) {
    var N = kernel._size[0];
    var kernel_new = m.ones(N, N+1);
    // console.log("kernel_new subset");
    // console.log(kernel_new.subset(m.index(m.range(0,N), m.range(1, N+1))));
    // console.log("kernel: ");
    // console.log(kernel);
    kernel_new.subset(
        m.index(
            m.range(0, N),
            m.range(1, N+1)
        ),
        kernel
    );
    // console.log("kernel_new: ");
    // console.log(kernel_new);
    return kernel_new;
}

function get_two_sigma_squared(sigma) {
    return m.multiply(
        -2.0,
        m.pow(
            sigma,
            2
        )
    );
}

function get_rbf_dist (dist_mtx, sigma) {
    //precompute two_sigma_squared
    var two_sigma_squared = get_two_sigma_squared(sigma);
    // console.log("two_sigma_squared: "+two_sigma_squared);
    var radial_dist_mtx = dist_mtx.map(function(dist) {
        var radial_dist = m.exp(
            m.divide(
                dist,
                two_sigma_squared
            )
        );
        // console.log("radial_dist: "+radial_dist);
        return radial_dist;
    });
    // console.log("radial_dist_mtx: ");
    // console.log(radial_dist_mtx);
    return radial_dist_mtx;
}

