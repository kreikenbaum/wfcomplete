"use strict";

function test(callback, thismany, mean, errorAllowed, assert) {
    var i;
    var sum = 0;
    for ( i = 0; i < thismany; i++ ) {
	sum += callback();
    }
    
    assert.ok(Math.abs(sum/thismany - mean) < errorAllowed, 
	      'failed, average: ' + (sum / thismany) 
	      + ', expected: ' + errorAllowed);
}
exports.test = test;
