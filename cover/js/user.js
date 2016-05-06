"use strict";

const NAME = 'user';
exports.DOC = 'what to do on traffic by the user';

const { setTimeout } = require("sdk/timers");

const _ = require('../lib/underscore-min.js');

const coverTraffic = require('./coverTraffic.js');
const debug = require('./debug.js');

const TIMEOUT = 110 * 1000;

let activeHosts = {};

// td: 
/** user starts loading url */
function loads(url) {
    debug.log("user: loads(" + url.spec + ")");
    if ( _.contains(activeHosts, url.host) ) { // has already started
	activeHosts[url.host].loadNext();
    } else {
	activeHosts[url.host] = new coverTraffic.CoverTraffic(url.spec);
	// td: removal code: better also watch endsload
	setTimeout(function() {
	    finish(url);
	}, TIMEOUT);
    }
}
exports.loads = url => loads(url);

function endsLoading(url) {
    finish(url);
}
exports.endsLoading = url => endsLoading(url);

/** tells covertraffic for {@code url.host} to {@code finish} up, deletes it */
function finish(url) {
    if ( activeHosts.hasOwnProperty(url.host) ) {
	activeHosts[url.host].finish();
	delete activeHosts[url.host];
    }
}
// // unused
// /** user ends loading url */
// function endsLoading(url) {
//     debug.traceLog(NAME + ': end user load ' + url.host);
//     if ( _.contains(activeHosts, url.host) ) {
// 	activeHosts = _.without(activeHosts, url.host);
//     }
//     if ( isIdle() ) {
// 	coverTraffic.stop();
//     }
//     debug.traceLog(activeHosts);
//     // td: if no sites are loading, switch mode
// };
// exports.endsLoading = url => endsLoading(url);

// // unused
// /** currently loading? */
// function isIdle() {
//     return _.isEmpty(activeHosts);
// };
// exports.isIdle = () => isIdle();
