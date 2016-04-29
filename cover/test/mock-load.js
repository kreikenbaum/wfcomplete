"use strict";
/**
* @fileoverview provides mock for {@code load.js}
*/
var params = [];

function getCount() {
    return params.length;
}
exports.getCount = getCount;

function getSum() {
    let sum = 0;
    params.forEach( (el) => { sum += el; } );
    return sum;
}
exports.getSum = getSum;



function http(toLoad) {
    params.push(toLoad);
}
exports.http = (toLoad) => http(toLoad);

function reset() {
    params = [];
}
exports.reset = reset;

function sized(size) {
    params.push(size);
}
exports.sized = (size) => sized(size);

