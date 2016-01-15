"use strict";

const NAME = 'user';
exports.DOC = 'traffic by the user';

const _ = require('./underscore-min.js');

const coverTraffic = require('./coverTraffic.js');
const debug = require('./debug.js');

// td: index by domain or by URL?
var activeHosts = [];

/** user starts loading url */
function loads(URL) {
    if ( ! _.contains(activeHosts, URL.host) ) {
	activeHosts.push(URL.host);
	coverTraffic.start();
	debug.traceLog(NAME + ': loads ' + URL.host);
	// td: should start cover traffic
    }
    debug.traceLog(activeHosts);
};
exports.loads = URL => loads(URL);

/** user ends loading url */
function endsLoading(URL) {
    debug.traceLog(NAME + ': end user load ' + URL.host);
    if ( _.contains(activeHosts, URL.host) ) {
	activeHosts = _.without(activeHosts, URL.host);
    }
    if ( isIdle() ) {
	coverTraffic.stop();
    }
    debug.traceLog(activeHosts);
    // td: if no sites are loading, switch mode
};
exports.endsLoading = URL => endsLoading(URL);

/** currently loading? */
function isIdle() {
    return _.isEmpty(activeHosts);
};
exports.isIdle = () => isIdle();
    
