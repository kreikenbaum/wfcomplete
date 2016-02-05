"use strict";

const {Cc, Ci} = require("chrome");

const pageMod = require("sdk/page-mod");

const debug = require("./debug.js");
const user = require("./user.js");
const watcher = require("js/watcher.js");

// 
watcher.register(user);

/** signals when page has finished loading */
pageMod.PageMod({
    include: "*",
    contentScript: "self.port.emit('loaded', document.location.href)",
    contentScriptWhen: "end",
    onAttach: function(worker) {
	worker.port.on("loaded", function(urlspec) {
	    debug.traceLog('loading of ' + urlspec + ' has stopped');
	    // td: refactor out
	    // td: see if replaceable by addon-module
	    var ioService = Cc["@mozilla.org/network/io-service;1"]
		.getService(Ci.nsIIOService);
	    user.endsLoading(ioService.newURI(urlspec, null, null));
	});
    }
});

exports.onUnload = function(reason) {
    watcher.unregister();
};
