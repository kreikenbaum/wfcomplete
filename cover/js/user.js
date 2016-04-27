"use strict";

const NAME = 'user';
exports.DOC = 'what to do on traffic by the user';

const { setTimeout } = require("sdk/timers");

const _ = require('../lib/underscore-min.js');

const coverTraffic = require('./coverTraffic.js');
const debug = require('./debug.js');

const TIMEOUT = 110 * 1000;

var activeHosts = {};

// td: timeout AND endsLoading
/** user starts loading url */
function loads(url) {
    debug.log("user: loads(" + url.spec + ")");
    if ( _.contains(activeHosts, url.host) ) { // has already started
	activeHosts[url.host].loadNext();
    } else {
	activeHosts[url.host] = new coverTraffic.CoverTraffic(url);
	// td: removal code: better also watch endsload
	setTimeout(function() {
	    delete activeHosts[url.host];
	}, TIMEOUT);
    }
}
exports.loads = url => loads(url);

function endsLoading(url) {
    debug.log('user: endsLoading('+url.host+') NOT IMPLEMENTED');
}
exports.endsLoading = url => endsLoading(url);
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
