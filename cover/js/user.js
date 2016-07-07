"use strict";
/**
* @fileoverview what to do on traffic by the user
*/
const { setTimeout } = require("sdk/timers");

const coverTraffic = require('./coverTraffic.js');

const TIMEOUT = 110 * 1000;

function User() {
    this.activeHosts = {};
}

/** user starts loading url, @return true if host has already had traffic */
User.prototype.loads = function(url) {
    console.log("user: loads(" + url.spec + ")");
    if ( url.host in this.activeHosts ) { // has already started
        // console.log(url.host + ' exists in dict: ' + JSON.stringify(activeHosts));
        this.activeHosts[url.host].loadNext();
        return true;
    } else {
        // console.log(url.host + ' is new in dict: ' + JSON.stringify(activeHosts));
        this.activeHosts[url.host] = new coverTraffic.CoverTraffic(url.spec);
        let that = this;
        setTimeout(function() {
            that.finish(url);
        }, TIMEOUT);
        return false;
    }
};

User.prototype.endsLoading = function(url) {
    return this.finish(url);
};

User.prototype.redirected = function(url) {
    // should be in activeHosts, otherwise fail loudly
    this.activeHosts[url.host].redirected();
};

/** tells covertraffic for {@code url.host} to {@code finish} up, deletes it,
 * @returns if action was taken */
User.prototype.finish = function(url) {
    if ( this.activeHosts.hasOwnProperty(url.host) ) {
        this.activeHosts[url.host].finish();
        delete this.activeHosts[url.host];
        return true;
    } else {
        return false;
    }
};

exports.User = User;
