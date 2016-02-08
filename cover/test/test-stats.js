"use strict";

var stats = require("../js/stats.js");

exports["test htmlMean"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 10000; i++ ) {
	sum += stats.htmlSize(1);
    }
    var mean = stats.htmlMean();
    assert.ok(Math.abs((sum / 10000) - mean) < 1000,
	      'mean: ' + mean + ' off:' + (sum/10000));
}

exports["test request length"] = function(assert) {
    var result = stats.requestLength();
    assert.ok(result <= 700, 'requestLength: ' + result + ' too big');
}

require("sdk/test").run(exports);
