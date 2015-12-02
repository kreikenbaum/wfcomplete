var pageMod = require("sdk/page-mod");

var DEBUG = true;
// td: remove, make empty, ...?
var FIXED_URLS = ["http://news.google.de/"]; // same as from torben? ;-)
//var TIMEOUT = 2000; // td: dynamically (and on someone loading something)

var candidates = FIXED_URLS.slice();

// user loads new pages
pageMod.PageMod({
    include: /.*/,
    contentScript: ['console.log("loading " + document.location.href);',
		    'self.port.emit("newPage", document.location.href.length)'],
    contentScriptWhen: "start",
    onAttach: function(worker) {
	worker.port.on("newPage", function(newLength) {
	    debugLog("new Page by user");
	    loadNext(newLength);
	});
    }
});

// listens for ajax calls
pageMod.PageMod({
    include: /.*/,
    contentScriptFile: "./listenForAjax.js",
    onAttach: function(worker) { // td: codup
	worker.port.on("ajax", function(newLength) {
	    debugLog("ajax loaded");
	    loadNext(newLength);
	});
    }
});
// td: loadNext() gets url, changes request size

// td: avoid triggering on file urls
// receives links from user page
pageMod.PageMod({
    include: /.*/,
    contentScriptFile: "./getLinks.js",
    onAttach: function(worker) {
	worker.port.on("links", function(JSONlinks) {
	    addToCandidates(JSON.parse(JSONlinks));
	});
    }
});

// attaches to an invisible tab, loads cover content there
pageWorker = require("sdk/page-worker").Page({
});

// td: limit size
// td: filter file:// urls
// add to possible urls to visit
function addToCandidates(array) {
    debugLog("addToCandidates("+array+")");
    candidates = candidates.concat(array);
}

function debugLog(toLog) {
    if ( DEBUG ) { console.log(toLog); }
}

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
