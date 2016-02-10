"use strict";

exports.DOC = 'loads pages over HTTP(S)';

// td: extract links for further cover traffic
// td: report size for cover traffic
// needs RequestPolicy to be disabled
/** @param toLoad load this url as cover (discarded afterwards) */
function http(toLoad) {
    require("sdk/page-worker").Page({
	contentURL: toLoad
    });
}
exports.http = (toLoad) => http(toLoad);

// td: use this
function sized(size) {
    http(require("./coverUrl.js").sized(size));
}
exports.sized = (size) => sized(size);
