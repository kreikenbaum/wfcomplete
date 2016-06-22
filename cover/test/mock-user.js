"use strict";
/**
* @fileoverview mock for user.js
*/
var loaded = [];

exports.loads = function(url) {
    loaded.push(url);
};

exports.loaded = loaded;
