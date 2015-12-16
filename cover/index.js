var {Cc, Ci} = require("chrome");
var pageMod = require("sdk/page-mod");
var { setTimeout } = require("sdk/timers");

var random = require("./random")
var url = require("./url")

const DEBUG = true;

var ignoreThese = [];

// listen to all requests, courtesy of stackoverflow.com/questions/21222873
httpRequestObserver = {
    observe: function(subject, topic, data) {
        if (topic == "http-on-modify-request") {
            var httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
            var uri = httpChannel.URI;

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


// needs RequestPolicy to be disabled
/** attaches to an invisible tab, loads cover content there, extracts
 * possibly new cover URLs from the loaded text */
pageWorker = require("sdk/page-worker").Page({
    contentScriptFile: "./getLinks.js",
    onAttach: function(worker) {
	worker.port.on("links", function(JSONlinks) {
	    debugLog("would add to links: " + JSONlinks);
	    // candidates.concat(JSON.parse(JSONlinks));
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
    // aka td: make nextUrl site-[(HTML|total)-]size-dependent
// td (currently Math.random)
    // get fitting object-url
    var nextUrl = url.sized(Math.floor(Math.random()*1000*1000));
    nextUrl += '?' + random.string(loadedUrl.length - nextUrl.length);
    ignoreThese.push(nextUrl);

    setTimeout(function() {
	ignoreThese.pop(nextUrl);
    }, 5);

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
