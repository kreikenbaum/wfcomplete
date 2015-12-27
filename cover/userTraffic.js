/** traffic by the user */
var _ = require('./underscore-min.js')

var debug = require('./debug.js')

// td: index by domain or by URL?
startTimes = {}

/** user starts loading url */
function start(URL) {
    startTimes[URL] = new Date();
    debug.log('userTraffic: should start periodic traffic');
}
exports.start = URL => start(URL);

/** user ends loading url */
function stop(URL) {
    delete startTimes[URL];
    debug.log('userTraffic: end do');
}
exports.stop = URL => stop(URL);

/** currently loading? */
function isIdle() {
    return _.isEmpty(startTimes);
}
exports.isIdle = () => isIdle();
    
