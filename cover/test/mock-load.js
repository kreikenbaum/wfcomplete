"use strict";
/**
* @fileoverview provides mock for {@code load.js}
*/
var count = 0;
var params = [];

function getCount() {
    return params.length;
}
exports.getCount = getCount;

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
