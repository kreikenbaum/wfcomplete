var {Cc, Ci} = require("chrome");
var pageMod = require("sdk/page-mod");

var DEBUG = true;
// td: sensible selection OR site-[(HTML|total)-]size-dependent OR ...?
var FIXED_URLS = ["http://news.google.de/"];

var candidates = FIXED_URLS.slice();
//td: use this
var page_requisites = [];

// listen to all requests, courtesy of stackoverflow.com/questions/21222873
httpRequestObserver = {
    observe: function(subject, topic, data) {
        if (topic == "http-on-modify-request") {
            // [...] do sth here, from answer:
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;
            //var domainloc = uri.host;

	    debugLog('observer: http request to ' + uri.spec);
	    loadNext(uri.spec.length); // td: maybe has length
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

// extracts (via getLinks) links from user page and appends to
// URL-candidates
pageMod.PageMod({
    include: /.*/,
    exclude: /(about:|file:).*/,
    contentScriptFile: "./getLinks.js",
    onAttach: function(worker) {
	worker.port.on("links", function(JSONlinks) {
	    candidates.concat(JSON.parse(JSONlinks));
	});
    }
});

// attaches to an invisible tab, loads cover content there
// needs RequestPolicy to be disabled
pageWorker = require("sdk/page-worker").Page({
});

function debugLog(toLog) {
    if ( DEBUG ) { console.log(toLog); }
}

// td: loadNext() gets url, changes request size
// loads next page in background
// adds random string to avoid caching and sets length of request
function loadNext(requestLength) {
    var url = next_url();
    url += '?' + randomString(requestLength - url.length);

    debugLog('loadNext: loading: ' + url);
    pageWorker.contentURL = url;
}

// td: scrape / onion plus some randomness
// yields the next url to visit
function next_url() {
    if ( candidates.length == 0 ) {
	candidates = FIXED_URLS.slice();
    }
    return candidates.pop();
}

// pseudo-random string of length length (at least one)
function randomString(length) {
    debugLog('randomString(' + length + ')');
    var iterations = 11 / length;
    var rands = [];
    for ( var i = 0; i < iterations; i++ ) {
	rands.push(Math.random().toString(36).substring(2));
    }
    var thisManyExtra = length - (iterations * 11);
    if ( thisManyExtra <= 0 ) { thisManyExtra = 1; }
    var extra = Math.random().toString(36).substring(2).slice(-thisManyExtra);

    return extra + rands.join('');
}

exports.onUnload = function(reason) {
    httpRequestObserver.unregister();
};
