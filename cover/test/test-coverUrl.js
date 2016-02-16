"use strict";

const coverUrl = require("../js/coverUrl.js");

exports["test contains"] = function(assert) {
    assert.ok(coverUrl.contains('somehost.com'), 'no contain');
};
exports["test contains"] = function(assert) {
    assert.ok(coverUrl.contains('somehost.com?size=1234'), 'no contain param');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};


exports["test sized"] = function(assert) {
    var result = coverUrl.sized(10)
    assert.ok(result === 'somehost.com:80/?size=10', 'failed: ' + result);
};


require("sdk/test").run(exports);
