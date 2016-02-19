"use strict";
/**
* @fileoverview loads pages over HTTP(S)
*/
const xhr = require("sdk/net/xhr");

const debug = require("./debug.js");

// needs RequestPolicy to be disabled
/** @param toLoad load this url as cover (discarded afterwards) */
function http(toLoad) {
    debug.log("load: http(" + toLoad + ")");
    var x = new xhr.XMLHttpRequest();
    x.addEventListener("load", reqListener);
    x.open("GET", toLoad);
    x.send();
//    debug.log("load: " + x.getRequestCount() + " active");
}
exports.http = (toLoad) => http(toLoad);

// td: use this
function sized(size) {
    debug.log("load: sized(" + size + ")");
    http(require("./coverUrl.js").sized(size));
}
exports.sized = (size) => sized(size);

function reqListener() {
    debug.log("load: response text[:4]: " + this.responseText.substr(0, 4));
}
