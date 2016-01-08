"use strict";

const NAME = 'user';
exports.DOC = 'traffic by the user';

const _ = require('./underscore-min.js');

const debug = require('./debug.js');

// td: index by domain or by URL?
var activeHosts = [];

/** user starts loading url */
function loads(URL) {
    if ( activeHosts.indexOf(URL.host) === -1 ) ) {
	activeHosts.push(URL.host);
	debug.log(NAME + ': loads ' + URL.host);
    }
    // td: should start cover traffic
    debug.traceLog(activeHosts);
};
exports.loads = URL => loads(URL);

/** user ends loading url */
function endsLoading(URL) {
    debug.log(NAME + ': end user load ' + URL.host);
    if ( _.contains(activeHosts, URL.host) {
	activeHosts.slice(activeHosts.indexOf(URL.host), 1);
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
    
