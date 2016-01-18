"use strict";

const _ = require('./underscore-min.js');
const {Cc, Ci} = require("chrome");

const pageMod = require("sdk/page-mod");
const { setTimeout } = require("sdk/timers");

const debug = require("./debug.js");
const coverUrl = require("./coverUrl.js");
const user = require("./user.js");

// listen to all requests, courtesy of stackoverflow.com/questions/21222873
var httpRequestObserver = {
    observe: function(subject, topic, data) {
	debug.traceLog('topic: ' + topic);
        if (topic == "http-on-modify-request") {
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;

	    if ( !coverUrl.contains(uri.spec) ) {
		debug.traceLog('observer: http request to ' + uri.spec);
		user.loads(uri);
//		loadNext(uri.spec);
	    } else {
		debug.log('ignoring request to ' + uri.spec);
	    }
        }
    },

    register: function() {
        var observerService = Cc["@mozilla.org/observer-service;1"]
            .getService(Ci.nsIObserverService);
        observerService.addObserver(this, "http-on-modify-request", false);
    },

    unregister: function() {
        var observerService = Cc["@mozilla.org/observer-service;1"]
            .getService(Ci.nsIObserverService);
        observerService.removeObserver(this, "http-on-modify-request");
    }
};
httpRequestObserver.register();

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
    httpRequestObserver.unregister();
};
