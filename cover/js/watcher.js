"use strict";
/**
* @fileoverview signals on user action
*/
const {Cc, Ci} = require("chrome");
const pageMod = require("sdk/page-mod");

const coverUrl = require("./coverUrl.js");

let callback;

/** page has finished loading */
let endObserver;
/** all requests */
// td: rename to httpObserver (also responses)
let httpRequestObserver = {
    observe: function(subject, topic, data) {
        if ( topic == "http-on-modify-request" ) {
	    // console.log(subject);
	    // console.log(JSON.stringify(subject));
	    // console.log('data:' + JSON.stringify(data));
            let httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            let uri = httpChannel.URI;

	    if ( !coverUrl.contains(uri.spec) ) {
		callback.loads(uri);
	    }
        } else if ( topic == "http-on-examine-response" ) {
            let httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            let uri = httpChannel.URI;

            if ( httpChannel.responseStatus !== 200 ) {
                console.log(' http Status: ' + httpChannel.responseStatus);
                callback.redirected(uri);
            }
        }
    },

    register: function() {
        let observerService = Cc["@mozilla.org/observer-service;1"]
            .getService(Ci.nsIObserverService);
        observerService.addObserver(this, "http-on-modify-request", false);
        observerService.addObserver(this, "http-on-examine-response", false);
    },

    unregister: function() {
        let observerService = Cc["@mozilla.org/observer-service;1"]
            .getService(Ci.nsIObserverService);
        observerService.removeObserver(this, "http-on-modify-request");
        observerService.removeObserver(this, "http-on-examine-response");
    }
};


/** start to watch traffic, callback to `user` object */
function register(user) {
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
