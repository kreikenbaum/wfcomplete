"use strict";

const NAME = 'user';
exports.DOC = 'what to do on traffic by the user';

const _ = require('../lib/underscore-min.js');

const coverTraffic = require('./coverTraffic.js');
coverTraffic.setLoader(require('./load.js'));
const debug = require('./debug.js');

var activeHosts = [];

/** user starts loading url */
function loads(URL) {
    if ( _.contains(activeHosts, URL.host) ) { // has already started
	coverTraffic.loadNext();
    } else {
	activeHosts.push(URL.host);
	coverTraffic.start();
    }
}
exports.loads = URL => loads(URL);

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
