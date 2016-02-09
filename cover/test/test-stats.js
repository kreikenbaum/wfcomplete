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

exports["test withProbability(>1)"] = function(assert) {
    assert.ok(stats.withProbability(1.1), 'off for prob > 1');
}
exports["test withProbability(<0)"] = function(assert) {
    assert.ok(!stats.withProbability(-0.1), 'off for prob < 0');
}
exports["test withProbability(0.5)"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 20; i += 1 ) {
	sum += (stats.withProbability(0.5) ? 1 : 0 );
    }
    assert.ok(Math.abs(sum - 10) <= 4, '(maybe) off for prob = 0.5');
}

require("sdk/test").run(exports);
