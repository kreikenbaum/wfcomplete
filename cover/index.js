"use strict";

const {Cc, Ci} = require("chrome");

const pageMod = require("sdk/page-mod");
const { setTimeout } = require("sdk/timers");

const debug = require("./debug.js");
const random = require("./random.js");
const stats = require("./stats.js");
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
	worker.port.on("loaded", function(urlspec) {
	    debug.traceLog('loading of ' + urlspec + ' has stopped');
	    // td: refactor out
	    var ioService = Cc["@mozilla.org/network/io-service;1"]
		.getService(Ci.nsIIOService);
	    user.endsLoading(ioService.newURI(urlspec, null, null));
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
    var nextUrl = coverUrl.sized(stats.htmlSize(1));

    // td: whole request size, not only url length
    // td: post for a post, get for a get
    var targetLength = (loadedUrl.length > 300)
	? stats.uniform(loadedUrl.length/2., loadedUrl.length *1.5)
	: stats.uniform(0, 300);
    var separator = nextUrl.indexOf('?') === -1 ) ? '?' : '&';
    nextUrl += separator + random.string(targetLength - nextUrl.length);

    // load next
    debug.log('loadNext: loading: ' + nextUrl);
    pageWorker.contentURL = nextUrl;
};

exports.onUnload = function(reason) {
    httpRequestObserver.unregister();
};
