"use strict";

const NAME = 'user';
exports.DOC = 'what to do on traffic by the user';

//td: rename to 'controller'?

const { setTimeout } = require("sdk/timers");
const { URL } = require("sdk/url");
//const {Cc, Ci} = require("chrome");
//const ioService = Cc["@mozilla.org/network/io-service;1"]
//		    .getService(Ci.nsIIOService);

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
	activeHosts[url.host] = new coverTraffic.CoverTraffic(url.host);
	// td: removal code: better also watch endsload
	setTimeout(function() {
	    delete activeHosts[url.host];
	}, TIMEOUT);
    }
}
exports.loads = url => loads(url);

/**
* @param {URL} url strips this
*
* @return {String} url without parameters and refs
*/
function stripParam(url) {
    console.log('stripParam(' + JSON.stringify(url) + ')');
    return URL(url.protocol + "//" + url.hostname + url.pathname);
}
exports.stripParam = url => stripParam(url); // testing

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
