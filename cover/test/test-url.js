var url = require("../url");

exports["test sizeof"] = function(assert) {
    assert.ok(url.sized(3) == 'http://mlsec.org/robots.txt', 'worlds');
}
exports["test sizeof2"] = function(assert) {
    assert.ok(url.sized(94) == 'http://mlsec.org/harry/api/nav_g.png', 'worlds');
}

require("sdk/test").run(exports);
