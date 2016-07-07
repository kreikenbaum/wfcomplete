"use strict";
/**
* @fileoverview what to do on traffic by the user
*/
const { setTimeout } = require("sdk/timers");

const coverTraffic = require('./coverTraffic.js');

const TIMEOUT = 110 * 1000;

let activeHosts = {};

/** user starts loading url, @return true if host has already had traffic */
function loads(url) {
    console.log("user: loads(" + url.spec + ")");
    if ( url.host in activeHosts ) { // has already started
	// console.log(url.host + ' exists in dict: ' + JSON.stringify(activeHosts));
	activeHosts[url.host].loadNext();
	return true;
    } else {
	// console.log(url.host + ' is new in dict: ' + JSON.stringify(activeHosts));
	activeHosts[url.host] = new coverTraffic.CoverTraffic(url.spec);
	setTimeout(function() {
	    finish(url);
	}, TIMEOUT);
	return false;
    }
}
exports.loads = url => loads(url);

function endsLoading(url) {
    return finish(url);
}
exports.endsLoading = url => endsLoading(url);

function redirected(url) {
    // should be in activeHosts, otherwise fail loudly
    activeHosts[url.host].redirected();
}
exports.redirected = () => redirected();

/** tells covertraffic for {@code url.host} to {@code finish} up, deletes it,
 * @returns if action was taken */
function finish(url) {
    if ( activeHosts.hasOwnProperty(url.host) ) {
	activeHosts[url.host].finish();
	delete activeHosts[url.host];
	return true;
    } else {
	return false;
    }
}

// // unused
// /** currently loading? */
// function isIdle() {
//     return _.isEmpty(activeHosts);
// };
// exports.isIdle = () => isIdle();
