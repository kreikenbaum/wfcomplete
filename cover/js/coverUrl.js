"use strict";
/**
* @fileoverview provides URLs for content of given sizes
*/
const Simple = require('sdk/simple-prefs');

const PROTOCOL = 'http://';

// traffic server preferences with update on change
let HOST = Simple.prefs['Traffic-HOST'];
Simple.on("Traffic-HOST", function() {
    HOST = Simple.prefs['Traffic-HOST'];
});
let PORT = Simple.prefs['Traffic-PORT'];
Simple.on("Traffic-PORT", function() {
    PORT = Simple.prefs['Traffic-PORT'];
});

function contains(url) {
    return url.indexOf(HOST) !== -1;
}
exports.contains = (url) => contains(url); // td: obsolete this, then rename
exports.includes = (url) => contains(url);

function sized(size) {
    return PROTOCOL + HOST + ":" + PORT + "/?size=" + Math.ceil(size);
}
exports.sized = (size) => sized(size);
