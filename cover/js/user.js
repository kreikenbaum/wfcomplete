"use strict";

const NAME = 'user';
exports.DOC = 'what to do on traffic by the user';

const { setTimeout } = require("sdk/timers");

const _ = require('../lib/underscore-min.js');

const coverTraffic = require('./coverTraffic.js');
const debug = require('./debug.js');
const load = require('./load.js');

const TIMEOUT = 110 * 1000;

var activeHosts = {};

// td: timeout or endsLoading
/** user starts loading url */
function loads(URL) {
    debug.log("user: loads(" + URL + ")");
    if ( _.contains(activeHosts, URL.host) ) { // has already started
	activeHosts[URL.host].loadNext();
    } else {
	activeHosts[URL.host] = new coverTraffic.CoverTraffic(load);
	// td: removal code: better also watch endsload
	setTimeout(function() {
	    delete activeHosts[URL.host];
	}, TIMEOUT);
    }
}
exports.loads = URL => loads(URL);

function endsLoading(URL) {
    debug.log('user: endsLoading('+URL.host+') NOT IMPLEMENTED');
};
exports.endsLoading = URL => endsLoading(URL);
// // unused
// /** user ends loading url */
// function endsLoading(URL) {
//     debug.traceLog(NAME + ': end user load ' + URL.host);
//     if ( _.contains(activeHosts, URL.host) ) {
// 	activeHosts = _.without(activeHosts, URL.host);
//     }
//     if ( isIdle() ) {
// 	coverTraffic.stop();
//     }
//     debug.traceLog(activeHosts);
//     // td: if no sites are loading, switch mode
// };
// exports.endsLoading = URL => endsLoading(URL);

// // unused
// /** currently loading? */
// function isIdle() {
//     return _.isEmpty(activeHosts);
// };
// exports.isIdle = () => isIdle();
