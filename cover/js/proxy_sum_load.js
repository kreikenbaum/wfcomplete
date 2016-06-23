"use strict";
/**
* @fileoverview loads pages over HTTP(S)
*/
const load = require("./load.js");
const stats = require("./stats.js");

// td: move overflow code to coverTraffic (or proxy object)
/** positive value: how much to subtract the next calls */
let overflow = 0;
/** minimum per load */
let MIN_SIZE = 160; 

/** @param {size int} loads resource of size {@code size} */
function sized(size) {
    console.log('proxy.sized(' + Math.round(size) + ')'); // | previous: '+overflow);
    if ( size <= MIN_SIZE ) {
        overflow += (MIN_SIZE - size);
    } else { // randomize how much to take from overflow
        let subtract = stats.uniform(0, Math.min(overflow, size - MIN_SIZE));
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
