"use strict";

const NAME = 'userTraffic';
exports.DOC = 'traffic by the user';

var _ = require('./underscore-min.js');

var debug = require('./debug.js');

// td: index by domain or by URL?
var startTimes = {};

/** user starts loading url */
function start(URL) {
    // td: filter: either just use host name or at least png, css, js
    if ( ! _.contains(startTimes, URL) ) {
	startTimes[URL] = new Date();
    }
    // td: should start periodic traffic
    debug.log(NAME + ': start load ' + URL);
};
exports.start = URL => start(URL);

/** user ends loading url */
function stop(URL) {
    debug.log(NAME + ': end user load ' + URL
	      + ' after ' + (new Date() - startTimes[URL]) + ' ms.');
    delete startTimes[URL];
};
exports.stop = URL => stop(URL);

/** currently loading? */
function isIdle() {
    return _.isEmpty(startTimes);
};
exports.isIdle = () => isIdle();
    
