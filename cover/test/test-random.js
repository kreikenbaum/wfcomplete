"use strict";

const random = require("../js/random.js");
const sumAvg = require("./sum-avg.js");

exports["test string len 94"] = function(assert) {
    var result = random.string(94);
    assert.equal(result.length, 94,
		 'failed, result ' + result
		 + ' has length ' + result.length);
};
exports["test string len 0"] = function(assert) {
    var result = random.string(0);
    assert.equal(result.length, 1,
		 'failed, result ' + result
		 + ' has length ' + result.length);
};
exports["test string repeat"] = function(assert) {
    var result = random.string(10);
    var resultb = random.string(10);
    assert.notEqual(result, resultb,
		    'failed, result ' + result
		    + ' equalled ' + resultb);
};
exports["test string len -5"] = function(assert) {
    var result = random.string(-5);
    assert.equal(result.length, 1,
		 'failed, result ' + result
		 + ' has length ' + result.length);
};
exports["test random uniform mean"] = function(assert) {
    sumAvg.test(random.uniform01, 100, 0.5, 0.2, assert);
};

require("sdk/test").run(exports);
