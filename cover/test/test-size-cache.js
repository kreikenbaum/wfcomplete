"use strict";

const SizeCache = require("../js/size-cache.js");
const stats = require("../js/stats.js");
const sumAvg = require("./sum-avg.js");

exports["test small html"] = function(assert) {
    assert.ok(SizeCache.htmlSize('http://google.com/'), 689,
              'wrong result: ' + SizeCache.htmlSize('http://www.google.com/'));
};

exports["test large html"] = function(assert) {
    assert.ok(SizeCache.htmlSize("http://www.amazon.com/"), 242532,
              'wrong result: ' + SizeCache.htmlSize('http://amazon.com/'));
};

exports["test unknown html"] = function(assert) {
    sumAvg.test(() => SizeCache.htmlSize('non-known string'),
                10000,
                stats.htmlMean(),
                0.2 * stats.htmlMean(),
                assert);
};

exports["test small num"] = function(assert) {
    assert.ok(SizeCache.numberEmbeddedObjects("https://www.google.de/"), 4,
              'wrong result: '
              + SizeCache.numberEmbeddedObjects('http://www.google.de/'));
};

exports["test large num"] = function(assert) {
    assert.ok(SizeCache.numberEmbeddedObjects("http://www.bbc.co.uk/"), 42,
              'wrong result: '
              + SizeCache.numberEmbeddedObjects("http://www.bbc.co.uk/"));
};

exports["test unknown num"] = function(assert) {
    sumAvg.test(() => SizeCache.numberEmbeddedObjects('non-known string'),
                1000,
		stats.numberEmbeddedObjectsMean(),
                Math.sqrt(stats.numberEmbeddedObjectsVar()),
                assert);
};

require("sdk/test").run(exports);
