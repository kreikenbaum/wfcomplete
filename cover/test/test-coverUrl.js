"use strict";

const coverUrl = require("../coverUrl.js");

const ROBOTS = 'http://mlsec.org/robots.txt';
const GOOGLE = 'http://www.google.com/';

exports["test contains(robots)"] = function(assert) {
    assert.ok(coverUrl.contains(ROBOTS), 'no contain');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};


exports["test coverFor - contained"] = function(assert) {
    var cover = coverUrl.coverFor(ROBOTS);
    assert.ok(coverUrl.contains(cover), 'cover: ' + cover + ' uncontained');
};


exports["test estimateLength(robots)"] = function(assert) {
    var len = coverUrl.estimateLength(ROBOTS, 27);
    assert.equal(len, 27, 'known length: ' + len);
};


exports["test isHTML(robots.txt)"] = function(assert) {
    assert.ok(! coverUrl.isHTML(ROBOTS), 'robots.txt diagnosed as HTML');
};
exports["test isHTML(google.com/)"] = function(assert) {
    assert.ok(coverUrl.isHTML(GOOGLE), GOOGLE + ' diagnosed as non-HTML');
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


exports["test sizeFor"] = function(assert) {
    var size = coverUrl.sizeFor('http://mlsec.org/sally/examples/jrc.zip');
    assert.ok(size == 0, 'failed sizeFor: ' + size);
};


require("sdk/test").run(exports);
