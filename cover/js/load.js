"use strict";
/**
* @fileoverview loads pages over HTTP(S)
*/
const Request = require("sdk/request").Request;

const debug = require("./debug.js");

// needs RequestPolicy to be disabled
/** @param toLoad load this url as cover (discarded afterwards) */
function http(toLoad) {
    debug.log("load: http(" + toLoad + ")");
    Request({
	url: toLoad,
	onComplete: function(response) {
	    debug.log("load: response: " + JSON.stringify(response));
	}
    }).get();
}
exports.http = (toLoad) => http(toLoad);

// td: use this
function sized(size) {
    debug.log("load: sized(" + size + ")");
    http(require("./coverUrl.js").sized(size));
}
exports.sized = (size) => sized(size);
