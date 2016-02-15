"use strict";
/**
* @fileoverview loads pages over HTTP(S)
*/

const debug = require("./debug.js");

// needs RequestPolicy to be disabled
/** @param toLoad load this url as cover (discarded afterwards) */
function http(toLoad) {
    require("sdk/page-worker").Page({
	contentURL: toLoad
    });
    debug.log("load: http(" + toLoad + ")");
}
exports.http = (toLoad) => http(toLoad);

// td: use this
function sized(size) {
    http(require("./coverUrl.js").sized(size));
}
exports.sized = (size) => sized(size);
