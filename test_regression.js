var fs = require('fs');
var m = require('mathjs');
var async = require('async');

var RVR = require('./rvr.js');


main(console.log);

function main(cb) {
    async.waterfall([
        get_file,
        make_training,
        build_model],
        cb
    )
}

function build_model(training, cb) {
    var rvr = new RVR({
        kernel: {
            type: "rbf",
            sigma: 0.02,
            normalize: true,
            bias: true
        },
        min_L_factor: .01,
        verbose: true
    });
    rvr.train({
        design: training.design.slice(0, training.design.length - 50),
        target: training.target.slice(0, training.design.length - 50)
    });
    var predictions = rvr.predict({
        design: training.design.slice(training.design.length - 50),
        target: training.target.slice(training.design.length - 50)
    });    
    console.log(predictions);    
    rvr.describe();
    console.log("N: "+rvr.N);
    cb();
}

function make_training(as_2d, cb) {
    // as_2d = as_2d.slice(as_2d.length - 40);
    var feature_length = as_2d[0].length;
    var design = [];
    var target = [];
    as_2d.forEach(function(data) {
        design.push(data.slice(0, feature_length - 1));
        target.push(+data.slice(feature_length - 1));
    });
    var training = {
        design: design,
        target: target
    }
            
    cb(null, training);
}

function get_file(cb) {
    fs.readFile('./datasets/housing.data', 'utf8', function(err, file) {
        if (err) {
            console.log(err);
            cb(err);
        } else {
            var lines = file.split('\n');
            var as_2d = lines.map(function(line) {
                var by_space = line.split(' ');
                var filtered = by_space.filter(function(item) {
                    return item != '';
                });
                return filtered;
            });
            as_2d.pop();
            cb(null, as_2d);
        }
    })
}