"use strict";
/**
* @fileoverview what to do on traffic by the user
*/
const Simple = require('sdk/simple-prefs');
const { setTimeout } = require("sdk/timers");

const Load = require('./proxy_sum_load.js');
const Stats = require("./stats.js");

const FACTOR = Simple.prefs.factor / 100;
const TIMEOUT = 110 * 1000;

function User() {
    this.activeHosts = [];
}

/** user starts loading url, @return true if host has already had traffic */
User.prototype.loads = function(url) {
    // console.log("user: loads(" + url.spec + ")");
    if ( contains(this.activeHosts, url.host) ) { // has already started
        for ( let i = ( Stats.withProbability( FACTOR % 1 ) ?
                        Math.ceil(FACTOR) :
                        Math.floor(FACTOR) ) ;
              i >= 1 ;
              i -= 1 ) {
            Load.sized(Stats.embeddedObjectSize());
        }
        return true;
    } else {
        this.activeHosts.push(url.host);
        Load.sized(Stats.htmlSize(FACTOR));
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

/** tells covertraffic for {@code url.host} to {@code finish} up, deletes it,
 * @returns if action was taken */
User.prototype.finish = function(url) {
    if ( contains(this.activeHosts, url.host) ) {
        this.activeHosts.splice(this.activeHosts.indexOf(url.host));
        return true;
    } else {
        return false;
    }
};

exports.User = User;

function contains(arr, el) {
    return arr.indexOf(el) !== -1;
}
exports.contains = contains; // for testing
