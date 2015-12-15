var random = require("../random");

exports["test string"] = function(assert) {
    assert.equal(random.string(94).length, 94,
	      'failed, result ' + random.string(94)
	      + ' has length ' + random.string(94).length);
}
exports["test string sized 0"] = function(assert) {
    assert.equal(random.string(0).length, 1,
	      'failed, result ' + random.string(0)
	      + ' has length ' + random.string(0).length);
}
exports["test string repeat"] = function(assert) {
    assert.notEqual(random.string(10), random.string(10).length,
	      'failed, result ' + random.string(10)
	      + ' equalled ' + random.string(10));
}

require("sdk/test").run(exports);
