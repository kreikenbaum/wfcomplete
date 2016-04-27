"use strict";

const stats = require("../js/stats.js");
const sumAvg = require("./sum-avg.js");

exports["test htmlSize mean"] = function(assert) {
    sumAvg.test(stats.htmlSize, 10000,
		stats.htmlMean(), 0.2 * stats.htmlMean(), assert);
};

exports["test htmlSize < limit"] = function(assert) {
    var result = stats.htmlSize();
    assert.ok(result < stats.HTML_TOP, 'htmlSize: ' + result + ' over cutoff');
};

exports["test htmlSize > 0"] = function(assert) {
    var result = stats.htmlSize();
    assert.ok(result > 0, 'htmlSize: ' + result + ' negative');
};

exports["test embeddedObjectSize mean"] = function(assert) {
    sumAvg.test(stats.embeddedObjectSize, 1000,
                stats.embeddedObjectSizeMean(),
                stats.embeddedObjectSizeStd(),
                assert);
};

exports["test embeddedObjectSizeMean < limit"] = function(assert) {
    var result = stats.embeddedObjectSize();
    assert.ok(result < stats.EMBEDDED_SIZE_TOP,
	      'embeddedObjectSize: ' + result + ' over cutoff');
};

exports["test embeddedObjectSize > 0"] = function(assert) {
    var result = stats.embeddedObjectSize();
    assert.ok(result > 0, 'embeddedObjectSize: ' + result + ' negative');
};

exports["test numberEmbeddedObjects > 0"] = function(assert) {
    var result = stats.numberEmbeddedObjects();
    assert.ok(result >= 0, 'numberEmbeddedObjects: ' + result + ' negative');
};

exports["test numberEmbeddedObjects < limit"] = function(assert) {
    var result = stats.numberEmbeddedObjects();
    assert.ok(result < stats.EMBEDDED_NUM_TOP,
	      'numberEmbeddedObjects: ' + result + ' over cutoff');
};

exports["test numberEmbeddedObjects is whole number"] = function(assert) {
    var result = stats.numberEmbeddedObjects();
    assert.ok(result % 1 === 0,
	      'numberEmbeddedObjects: ' + result + ' not a whole number');
};

exports["test numberEmbeddedObjects mean"] = function(assert) {
    sumAvg.test(stats.numberEmbeddedObjects, 1000,
		stats.numberEmbeddedObjectsMean(),
                Math.sqrt(stats.numberEmbeddedObjectsVar()),
		assert);
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
    for ( var i = 0; i < 100; i += 1 ) {
	sum += stats.withProbability(0.5); // js type coercion to 0 or 1 ;-)
    }
    assert.ok(Math.abs(sum - 50) <= 10, '(maybe) off for prob = 0.5');
};

require("sdk/test").run(exports);
