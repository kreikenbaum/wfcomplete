var stats = require("../stats");

exports["test mean"] = function(assert) {
    var sum = 0;
    for ( var i = 0; i < 10000; i++ ) {
	sum += stats.htmlSize(1);
    }
    assert.ok(Math.abs((sum / 10000) - stats.htmlMean()) < 500,
	      'mean: ' + stats.htmlMean() + ' off:' + (sum/10000));
}

require("sdk/test").run(exports);
