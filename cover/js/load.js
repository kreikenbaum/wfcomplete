"use strict";
/**
* @fileoverview loads pages over HTTP(S)
*/
const xhr = require("sdk/net/xhr");

const coverUrl = require("./coverUrl.js");

/** @param toLoad load this url as cover (discarded afterwards) */
function http(toLoad) {
//    console.log("load: http(" + toLoad + ")");
    let x = new xhr.XMLHttpRequest();
//    x.addEventListener("load", reqListener);
    x.open("GET", toLoad);
    x.send();
}
exports.http = (toLoad) => http(toLoad);

/** @param {size int} loads resource of size {@code size} */
function sized(size) {
    console.log("load: sized(" + size + ")");
    http(coverUrl.sized(size));
}
exports.sized = (size) => sized(size);

// function reqListener() {
//     debug.log("load: response length: " + this.responseText.length);
// }
// [:4]: " + this.responseText.substr(0, 4) +
