"use strict";

const user = require("js/user.js");
const watcher = require("js/watcher.js");

watcher.register(user);

exports.onUnload = function(reason) {
    watcher.unregister();
};
