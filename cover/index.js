"use strict";

const {Cc, Ci} = require("chrome");

const pageMod = require("sdk/page-mod");
const { setTimeout } = require("sdk/timers");

const debug = require("./debug.js");
const random = require("./random.js");
const stats = require("./stats.js");
const coverUrl = require("./coverUrl.js");
const userTraffic = require("./userTraffic.js");

// listen to all requests, courtesy of stackoverflow.com/questions/21222873
var httpRequestObserver = {
    observe: function(subject, topic, data) {
	debug.traceLog('topic: ' + topic);
        if (topic == "http-on-modify-request") {
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
	    // debug.traceLog('channel: ' + httpChannel);
            var uri = httpChannel.URI;
	    // debug.traceLog('uri: ' + uri);

	    if ( !coverUrl.contains(uri.spec) ) {
		debug.traceLog('observer: http request to ' + uri.spec);
		userTraffic.start(uri.spec);
		loadNext(uri.spec);
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
	worker.port.on("loaded", function(url) {
	    debug.log('loading of ' + url + ' has stopped');
	    userTraffic.stop(url);
	});
    }
});

// needs RequestPolicy to be disabled
/** attaches to an invisible tab, loads cover content there, extracts
 * possibly new cover URLs from the loaded text */
var pageWorker = require("sdk/page-worker").Page({
    contentScriptFile: "./getLinks.js",
    onAttach: function(worker) {
	worker.port.on("links", function(JSONlinks) {
	    debug.log("td: add to cover links: " + JSONlinks);
	    // candidates.concat(JSON.parse(JSONlinks));
	});
    }
});

// loads next page in background
// adds random string to avoid caching and sets length of request
function loadNext(loadedUrl) {
    debug.traceLog('loadNext(' + loadedUrl + ')');

    // get fitting object-url
    // debug.traceLog(coverUrl);
    var nextUrl = coverUrl.coverFor(loadedUrl);
    // td: this adds '?' even if the url already contains it
    nextUrl += '?' + random.string(loadedUrl.length - nextUrl.length);

    // load next
    debug.log('loadNext: loading: ' + nextUrl);
    pageWorker.contentURL = nextUrl;
};

exports.onUnload = function(reason) {
    httpRequestObserver.unregister();
};
