"use strict";

const User = require("js/user.js");
const Watcher = require("js/watcher.js");

let watcher = new Watcher.Watcher(new User.User());

exports.onUnload = function(reason) {
    watcher.unregister();
};
