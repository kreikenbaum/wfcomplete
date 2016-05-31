"use strict";

/** mock url, call constructor with just a hostname */
exports.URL = function(hostname) {
    this.host = hostname;
    this.href = hostname;
    this.spec = hostname;
};
exports.URL.prototype.toString = function() {
    return JSON.stringify(this);
};
