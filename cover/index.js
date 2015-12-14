var {Cc, Ci} = require("chrome");
var pageMod = require("sdk/page-mod");

var random = require("./random")
var url = require("./url")

var DEBUG = true;

var ignoreThese = [];


// listen to all requests, courtesy of stackoverflow.com/questions/21222873
httpRequestObserver = {
    observe: function(subject, topic, data) {
        if (topic == "http-on-modify-request") {
            // [...] do sth here, from answer:
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;
            //var domainloc = uri.host;

	    if ( ignoreThese.indexOf(uri.spec) == -1 ) {
		debugLog('observer: http request to ' + uri.spec);
		loadNext(uri.spec);
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

// attaches to an invisible tab, loads cover content there
// needs RequestPolicy to be disabled
pageWorker = require("sdk/page-worker").Page({
    contentScriptFile: "./getLinks.js",
    onAttach: function(worker) {
	worker.port.on("links", function(JSONlinks) {
	    candidates.concat(JSON.parse(JSONlinks));
	});
    }
});

function debugLog(toLog) {
    if ( DEBUG ) { console.log(toLog); }
}

// loads next page in background
// adds random string to avoid caching and sets length of request
function loadNext(loadedUrl) {
    // determine size of next request
// td (currently Math.random)
    // get fitting object-url
    // td: make nextUrl site-[(HTML|total)-]size-dependent
    var nextUrl = url.sized(Math.floor(Math.random()*1000*1000));
    nextUrl += '?' + random.string(loadedUrl.length - nextUrl.length);
    ignoreThese.push(nextUrl);

    debugLog('loadNext: loading: ' + nextUrl);
    pageWorker.contentURL = nextUrl;
}

// td: scrape / onion plus some randomness
// yields the next url to visit
function next_url(length) {
    if ( candidates.length == 0 ) {
	candidates = FIXED_URLS.slice();
    }
    return candidates.pop();
}


exports.onUnload = function(reason) {
    httpRequestObserver.unregister();
};
