var url = require("../url");

exports["test sized"] = function(assert) {
    assert.ok(url.sized(3) == 'http://mlsec.org/robots.txt', 'failed');
}
exports["test sized2"] = function(assert) {
    assert.ok(url.sized(94) == 'http://mlsec.org/harry/api/nav_g.png',
	      'failed, result ' + url.sized(94));
}
exports["test sized too big"] = function(assert) {
    assert.ok(url.sized(123456789) == 'http://mlsec.org/sally/examples/jrc.zip', 'failed');
}

require("sdk/test").run(exports);
