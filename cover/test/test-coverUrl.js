"use strict";

const coverUrl = require("../coverUrl.js");
const stats = require("../stats.js");

const ROBOTS = 'http://mlsec.org/robots.txt';
const GOOGLE = 'http://www.google.com/';

exports["test contains(robots)"] = function(assert) {
    assert.ok(coverUrl.contains(ROBOTS), 'no contain');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};




exports["test sized"] = function(assert) {
    assert.ok(coverUrl.sized(3) == 'http://mlsec.org/robots.txt', 'failed');
};
exports["test sized2"] = function(assert) {
    assert.ok(coverUrl.sized(94) == 'http://mlsec.org/harry/api/nav_g.png',
	      'failed, result ' + coverUrl.sized(94));
};
exports["test sized too big"] = function(assert) {
    assert.ok(coverUrl.sized(123456789) == 'http://mlsec.org/sally/examples/jrc.zip', 'failed');
};


require("sdk/test").run(exports);
// unbenutzt
/*
exports["test coverFor - contained"] = function(assert) {
    var cover = coverUrl.coverFor(ROBOTS);
    assert.ok(coverUrl.contains(cover), 'cover: ' + cover + ' uncontained');
};


exports["test estimateLength(html)"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 10000; i++ ) {
	sum += coverUrl.estimateLength('http://google.com/');
    }
    var mean = stats.htmlMean();
    assert.ok(Math.abs((sum / 10000) - mean) < 1000,
	      'mean: ' + mean + ' off:' + (sum/10000));
};

exports["test estimateLength(non-html)"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 10000; i++ ) {
	sum +=coverUrl.estimateLength('http://mlsec.org/sally/examples/jrc.zip');
    }
    var mean = stats.embeddedObjectMean();
    assert.ok(Math.abs((sum / 10000) - mean) < 1000,
	      'mean: ' + mean + ' off:' + (sum/10000));
};


exports["test isHTML(robots.txt)"] = function(assert) {
    assert.ok(! coverUrl.isHTML(ROBOTS), 'robots.txt diagnosed as HTML');
};
exports["test isHTML(google.com/)"] = function(assert) {
    assert.ok(coverUrl.isHTML(GOOGLE), GOOGLE + ' diagnosed as non-HTML');
};


exports["test sizeFor"] = function(assert) {
    var size = coverUrl.sizeFor('http://mlsec.org/sally/examples/jrc.zip');
    assert.ok(size == 0, 'failed sizeFor: ' + size);
};

*/
