"use strict";

const stats = require("../js/stats.js");
const sumAvg = require("./sum-avg.js");

exports["test htmlMean"] = function(assert) {
    sumAvg.test(stats.htmlSize, 10000,
		stats.htmlMean(), 0.2 * stats.htmlMean(), assert);
};

exports["test htmlSize > 0"] = function(assert) {
    var result = stats.htmlSize();
    assert.ok(result > 0, 'htmlSize: ' + result + ' negative');
};

exports["test embeddedMean"] = function(assert) {
    sumAvg.test(stats.embeddedObjectSize, 20000,
		stats.embeddedObjectMean(), 0.3 * stats.embeddedObjectMean(),
		assert);
};

exports["test embeddedObjectSize > 0"] = function(assert) {
    var result = stats.embeddedObjectSize();
    assert.ok(result > 0, 'embeddedObjectSize: ' + result + ' negative');
};

exports["test embeddedObjectNumber > 0"] = function(assert) {
    var result = stats.numberEmbeddedObjects();
    assert.ok(result >= 0, 'numberEmbeddedObjects: ' + result + ' negative');
};

exports["test embeddedObjectNumber is whole number"] = function(assert) {
    var result = stats.numberEmbeddedObjects();
    assert.ok(result % 1 === 0,
	      'numberEmbeddedObjects: ' + result + ' not a whole number');
};

exports["test request length"] = function(assert) {
    var result = stats.requestLength();
    assert.ok(result <= 700, 'requestLength: ' + result + ' too big');
};

exports["test request length >0"] = function(assert) {
    var result = stats.requestLength();
    assert.ok(result >0, 'requestLength: ' + result + ' <= 0');
};

exports["test withProbability(>1)"] = function(assert) {
    assert.ok(stats.withProbability(1.1), 'off for prob > 1');
};
exports["test withProbability(<0)"] = function(assert) {
    assert.ok(!stats.withProbability(-0.1), 'off for prob < 0');
};
exports["test withProbability(0.5)"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 20; i += 1 ) {
	sum += (stats.withProbability(0.5) ? 1 : 0 );
    }
    assert.ok(Math.abs(sum - 10) <= 4, '(maybe) off for prob = 0.5');
};

require("sdk/test").run(exports);
