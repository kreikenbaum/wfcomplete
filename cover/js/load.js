"use strict";

exports.DOC = 'loads pages over HTTP(S)';

// l8r: keep track, extract links for further cover traffic, etc
// l8r: mock this to test that stuff is loaded
// needs RequestPolicy to be disabled
/** loads stuff over http */
function http(toLoad) {
    require("sdk/page-worker").Page({
	contentURL: toLoad
    });
};
exports.http = toLoad => http(toLoad);


