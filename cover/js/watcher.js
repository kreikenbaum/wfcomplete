"use strict";

exports.DOC = 'signals on user action';

const {Cc, Ci} = require("chrome");
const coverUrl = require("../coverUrl.js");
const debug = require("../debug.js");

var callback;

// listen to all requests, courtesy of stackoverflow.com/questions/21222873
var httpRequestObserver = {
    observe: function(subject, topic, data) {
	debug.traceLog('topic: ' + topic);
        if (topic == "http-on-modify-request") {
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;

	    if ( !coverUrl.contains(uri.spec) ) {
		debug.traceLog('observer: http request to ' + uri.spec);
		callback.loads(uri);
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

/** start to watch traffic, callback to `user` object */
function register(user) {
    callback = user;
    httpRequestObserver.register();
};
exports.register = user => register(user);

/** shuts down watcher */
function unregister() {
    httpRequestObserver.unregister();
    callback = null;
};    
exports.unregister = unregister;


