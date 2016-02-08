"use strict";

const coverUrl = require("../js/coverUrl.js");
const stats = require("../js/stats.js");

const ROBOTS = 'http://mlsec.org/robots.txt';

exports["test contains(robots)"] = function(assert) {
    assert.ok(coverUrl.contains(ROBOTS), 'no contain');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};


exports["test sized negative"] = function(assert) {
    var result = coverUrl.sized(-30).split('?')[0];
    assert.ok(result === ROBOTS, 'failed: ' + result);
};
exports["test sized tiny"] = function(assert) {
    var result = coverUrl.sized(3).split('?')[0];
    assert.ok(result === ROBOTS, 'failed: ' + result);
};
exports["test sized fix"] = function(assert) {
    var result = coverUrl.sized(94).split('?')[0];
    assert.ok(result === 'http://mlsec.org/harry/api/nav_g.png',
	      'failed, result ' + coverUrl.sized(94));
};
exports["test sized too big"] = function(assert) {
    var result = coverUrl.sized(123456789).split('?')[0];
    assert.ok(result === 'http://mlsec.org/sally/examples/jrc.zip',
	      'failed: ' + result);
};


require("sdk/test").run(exports);
