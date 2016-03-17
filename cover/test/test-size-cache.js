"use strict";

const SizeCache = require("../js/size-cache.js");
const stats = require("../js/stats.js");
const sumAvg = require("./sum-avg.js");

exports["test small"] = function(assert) {
    assert.ok(SizeCache.htmlSize('http://www.google.com/'), 491, 
              'wrong result: ' + SizeCache.htmlSize('http://www.google.com/'));
};

exports["test middle"] = function(assert) {
    assert.ok(SizeCache.htmlSize("http://amazon.com/"), Math.round(2705), 
              'wrong result: ' + SizeCache.htmlSize('http://amazon.com/'));
};

exports["test large"] = function(assert) {
    assert.ok(SizeCache.htmlSize("http://www.qq.com/"), Math.round(14906), 
              'wrong result: ' + SizeCache.htmlSize('http://www.google.com/'));
};

exports["test unknown"] = function(assert) {
    sumAvg.test(function() { return SizeCache.htmlSize('non-known string')}, 
                1000,
		stats.htmlMean(), 0.3 * stats.htmlMean(), assert);
};

require("sdk/test").run(exports);
