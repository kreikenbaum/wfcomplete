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
exports["test random uniform mean"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 10000; i++ ) {
	sum += random.random();
    }
    
    assert.ok(Math.abs(sum - 5000) < 200, 'failed, average: ' + (sum / 10000));
}



require("sdk/test").run(exports);
