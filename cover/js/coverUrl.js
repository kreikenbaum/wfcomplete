"use strict";
/**
* @fileoverview provides URLs for content of given sizes
*/
// td: preference for host name
const HOST = 'somehost.com';

function contains(url) {
    return url.indexOf(HOST) !== -1;
}
exports.contains = (url) => contains(url); // td: obsolete this, then rename
exports.includes = (url) => contains(url);

function sized(size) {
    return HOST + "/?size=" + size
}
exports.sized = (size) => sized(size);
