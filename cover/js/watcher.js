"use strict";
/**
* @fileoverview signals on user action
*/
const {Cc, Ci} = require("chrome");
const pageMod = require("sdk/page-mod");

const coverUrl = require("./coverUrl.js");

function Watcher(callback) {
    this.callback = callback;
    this.endObserver = createPageMod(this.callback);
    this.register();
}

/** all requests */
// td: rename to httpObserver (also responses)
Watcher.prototype.observe = function(subject, topic, data) {
    if ( topic == "http-on-modify-request" ) {
        // console.log(subject);
        // console.log(JSON.stringify(subject));
        // console.log('data:' + JSON.stringify(data));
        let httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
        let uri = httpChannel.URI;

        if ( !coverUrl.contains(uri.spec) ) {
            this.callback.loads(uri);
        }
    } else if ( topic == "http-on-examine-response" ) {
        let httpChannel = subject.QueryInterface(Ci.nsIHttpChannel);
        let uri = httpChannel.URI;

        if ( httpChannel.responseStatus !== 200 ) {
            console.log('http Status: ' + httpChannel.responseStatus);
            this.callback.redirected(uri);
        }
    }
};

Watcher.prototype.register = function() {
    let observerService = Cc["@mozilla.org/observer-service;1"]
        .getService(Ci.nsIObserverService);
    observerService.addObserver(this, "http-on-modify-request", false);
    observerService.addObserver(this, "http-on-examine-response", false);
};

Watcher.prototype.unregister = function() {
    let observerService = Cc["@mozilla.org/observer-service;1"]
        .getService(Ci.nsIObserverService);
    observerService.removeObserver(this, "http-on-modify-request");
    observerService.removeObserver(this, "http-on-examine-response");
};

exports.Watcher = Watcher;

/** watches for end of page load (all images etc loaded) */
function createPageMod(callback) {
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
