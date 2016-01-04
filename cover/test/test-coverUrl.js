var coverUrl = require("../coverUrl");

exports["test sized"] = function(assert) {
    assert.ok(coverUrl.sized(3) == 'http://mlsec.org/robots.txt', 'failed');
}
exports["test sized2"] = function(assert) {
    assert.ok(coverUrl.sized(94) == 'http://mlsec.org/harry/api/nav_g.png',
	      'failed, result ' + coverUrl.sized(94));
}
exports["test sized too big"] = function(assert) {
    assert.ok(coverUrl.sized(123456789) == 'http://mlsec.org/sally/examples/jrc.zip', 'failed');
}

exports["test lengthOf(robots)"] = function(assert) {
    var len = coverUrl.lengthOf('http://mlsec.org/robots.txt', 27);
    assert.equal(len, 27, 'known length: ' + len);
}

require("sdk/test").run(exports);
