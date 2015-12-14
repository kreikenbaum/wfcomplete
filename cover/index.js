var {Cc, Ci} = require("chrome");
var pageMod = require("sdk/page-mod");

var random = require("./random")

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
	    loadNext(uri.spec);
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

// td: loadNext() gets url, changes request size
// loads next page in background
// adds random string to avoid caching and sets length of request
function loadNext(loadedUrl) {
    // determine size of next request
// td
    // get fitting object-url
// td    
    
    var url = next_url();
    url += '?' + random.string(loadedUrl.length - url.length);

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


exports.onUnload = function(reason) {
    httpRequestObserver.unregister();
};
