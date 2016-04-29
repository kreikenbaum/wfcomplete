"use strict";

function test(callback, thismany, mean, errorAllowed, assert) {
    var i;
    var sum = 0;
    for ( i = 0; i < thismany; i++ ) {
	sum += callback();
    }
    
    assert.ok(Math.abs(sum/thismany - mean) < errorAllowed, 
	      'failed, off mean by: ' + (sum / thismany - mean) +
	      ', allowed: ' + errorAllowed);
}
exports.test = test;
