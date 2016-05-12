"use strict";
/**
* @fileoverview signals on user action
*/
const {Cc, Ci} = require("chrome");
const pageMod = require("sdk/page-mod");

const coverUrl = require("./coverUrl.js");

var callback;

/** page has finished loading */
var endObserver;
/** all requests */
var httpRequestObserver = {
    observe: function(subject, topic, data) {
        if (topic == "http-on-modify-request") {
	    // console.log(subject);
	    // console.log('data:' + JSON.stringify(data));
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;

	    if ( !coverUrl.contains(uri.spec) ) {
		callback.loads(uri);
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


/** start to watch traffic, callback to `user` object */
function register(user) {
    console.log('register');
    callback = user;
    endObserver = createPageMod();
    httpRequestObserver.register();
}
exports.register = user => register(user);

/** shuts down watcher */
function unregister() {
    httpRequestObserver.unregister();
    // how to delete endObserver?
}
exports.unregister = unregister;


/** watches for end of page load (all images etc loaded) */
function createPageMod() {
    return pageMod.PageMod({
	include: "*",
	contentScript: "self.port.emit('loaded', document.location.host)",
	contentScriptWhen: "end",
	onAttach: function(worker) {
	    worker.port.on("loaded", function(host) {
		callback.endsLoading(host);
	    });
	}
    });
}
