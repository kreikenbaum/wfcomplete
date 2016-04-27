"use strict";

const Simple = require('sdk/simple-prefs');
const { URL } = require("sdk/url");

const SizeCache = require("../js/size-cache.js");
//const stats = require("../js/stats.js");
//const sumAvg = require("./sum-avg.js");

const TEST = URL('http://www.google.com/search');

exports["test small html"] = function(assert) {
    assert.equal(SizeCache.htmlSize('http://google.com/'), 689);
};

exports["test small html with param"] = function(assert) {
    assert.equal(SizeCache.htmlSize('http://google.com/?testparam'), 689);
};

exports["test large html"] = function(assert) {
    assert.equal(SizeCache.htmlSize("http://www.amazon.com/"), 242532);
};

// td: use htmlStd,
exports["test unknown html"] = function(assert) {
    assert.throws(() => SizeCache.htmlSize('http://non-known.string'),
                  /no entries matching/,
		  "unknown string wrong behavior");
};

// td: heed strategies
// exports["test passive html"] = function(assert) {
//     let cache = Simple.prefs.bloom;
//     Simple.prefs.bloom=false;
//     sumAvg.test(() => SizeCache.htmlSize('http://google.com/'),
//                 1000,
//                 stats.htmlMean(),
//                 0.2 * stats.htmlMean(),
//                 assert);
//     Simple.prefs.bloom=cache;
// };

exports["test small html max"] = function(assert) {
    assert.equal(SizeCache.htmlSizeMax('http://google.com/'), 955);
};

exports["test small num"] = function(assert) {
    assert.equal(SizeCache.numberEmbeddedObjects("https://www.google.de/"), 4);
};

exports["test large num"] = function(assert) {
    assert.equal(SizeCache.numberEmbeddedObjects("http://www.bbc.co.uk/"), 42);
};

exports["test unknown num"] = function(assert) {
    assert.throws(() =>
		  SizeCache.numberEmbeddedObjects('http://non-known.string'),
                  /no entries matching/,
		  "unknown string wrong behavior");
};

// td: set strategy
// exports["test passive num"] = function(assert) {
//     let cache = Simple.prefs.bloom;
//     Simple.prefs.bloom=false;
//     sumAvg.test(() => SizeCache.numberEmbeddedObjects('http://google.com/'),
//                 1000,
// 		stats.numberEmbeddedObjectsMean(),
//                 Math.sqrt(stats.numberEmbeddedObjectsVar()),
//                 assert);
//     Simple.prefs.bloom=cache;
// };

exports["test stripParam"] = function(assert) {
    //    console.log("url: " + TEST.href);
    assert.equal(SizeCache.stripParamHref(TEST.href + '?asdf'), TEST.href);
    assert.equal(SizeCache.stripParamHref(URL(TEST.href + '?asdf')), TEST.href);
    assert.equal(SizeCache.stripParamHref(TEST), TEST.href);
    assert.equal(SizeCache.stripParamHref(TEST.href + '#test'), TEST.href);
    assert.equal(SizeCache.stripParamHref("http://passport.weibo.com/visitor/visitor"), "http://passport.weibo.com/visitor/visitor");
};

require("sdk/test").run(exports);
