var RVR = require('./rvr.js');


console.log("running a test");

var params = {
    kernel: {
        type: "rbf",
        sigma: 1.0
    }
};

var params_2 = {
    kernel: {
        type: "rbf",
        sigma: 0.5,
        normalize: false,
        bias: false
    },
};

var params_3 = {
    kernel: {
        type: "rbf",
        sigma: 1,
        bias: true
    },
    method: "classification"
};

var design = [
    [1, 2, 3],
    [4, 5, 6],
    [7,8,9]
];
var target = [0.1, 4];

var class_target = [1, 1, 0];

// var rvr = new RVR(params);
// rvr.describe();

// var rvr_2 = new RVR(params_2);
// rvr_2.describe();
// rvr_2.train({design: design, target: target});
// var predictions = rvr_2.predict({
    // design: [
        // [1, 2, 3],
        // [4, 5, 6],
    // ],
    // target: [4.3, 3]
// });
// console.log(predictions);

var rvr_3 = new RVR(params_3);
rvr_3.train({design: design, target: class_target});
var pred_3 = rvr_3.predict({
    design: [
        [0, 2, 3]
    ],
    target: [1]
});
console.log(pred_3);


// var rvr_3 = new RVR(params_3);
// rvr_3.describe();