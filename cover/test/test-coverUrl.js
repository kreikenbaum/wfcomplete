"use strict";

const coverUrl = require("../coverUrl.js");
const stats = require("../stats.js");

const ROBOTS = 'http://mlsec.org/robots.txt';

exports["test contains(robots)"] = function(assert) {
    assert.ok(coverUrl.contains(ROBOTS), 'no contain');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};


exports["test sized negative"] = function(assert) {
    assert.ok(coverUrl.sized(-30) == ROBOTS, 'failed');
};
exports["test sized tiny"] = function(assert) {
    assert.ok(coverUrl.sized(3) == ROBOTS, 'failed');
};
exports["test sized fix"] = function(assert) {
    assert.ok(coverUrl.sized(94) == 'http://mlsec.org/harry/api/nav_g.png',
	      'failed, result ' + coverUrl.sized(94));
};
exports["test sized too big"] = function(assert) {
    assert.ok(coverUrl.sized(123456789) == 'http://mlsec.org/sally/examples/jrc.zip', 'failed');
};


require("sdk/test").run(exports);
