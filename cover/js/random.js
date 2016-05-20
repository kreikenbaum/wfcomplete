"use strict";

exports.DOC = 'random.string and crypto-secure randomness';

const { Cu } = require("chrome");
Cu.importGlobalProperties(["crypto"]); /* globals crypto */

// courtesy of stackoverflow.com/questions/1349404
// does not need strong randomness
/** returns pseudo-random string of length length */
function string(length) {
    if ( length <= 0 ) { length = 1; }
    var text = "";
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for( var i=0; i < length; i++ )
        text += possible.charAt(Math.floor(Math.random() * possible.length));

    return text;
}
exports.string = (length) => string(length);

/**
* uniform secure random number in [0, 1)
*/
function uniform01() {
    var buffer = new ArrayBuffer(8);
    var ints = new Int8Array(buffer);
    crypto.getRandomValues(ints);

    // Set the sign (ints[7][7]) to 0, and the exponent (ints[7][6] -
    // [6][5]) to the right size (all ones except for the highest bit)
    ints[7] = 63;
    ints[6] |= 0xf0;

    var float = new Float64Array(buffer)[0] - 1;
    return float;
}
exports.uniform01 = uniform01;
