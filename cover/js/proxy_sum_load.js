"use strict";
/**
* @fileoverview loads pages over HTTP(S), equalizes negative byte counts
*/
const load = require("./load.js");
const stats = require("./stats.js");
const Simple = require('sdk/simple-prefs');

// td: move overflow code to coverTraffic (or proxy object)
/** positive value: how much to subtract the next calls */
let overflow = 0;

/** @param {size int} loads resource of size {@code size} */
function sized(size) {
    console.log('proxy.sized(' + Math.round(size) + ')'); // | previous: '+overflow);
    if ( size <= Simple.prefs.min_size ) {
        overflow += (Simple.prefs.min_size - size);
    } else { // randomize how much to take from overflow
        let subtract = stats.uniform(0, Math.min(overflow,
                                                 size - Simple.prefs.min_size));
        overflow -= subtract;
        size -= subtract;
    }
    load.sized(size);
}
exports.sized = (size) => sized(size);

// function reqListener() {
//     debug.log("load: response length: " + this.responseText.length);
// }
// [:4]: " + this.responseText.substr(0, 4) +
