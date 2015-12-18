/** traffic by the user */
var _ = require('./underscore-min.js')

// td: index by domain or by URL?
startTimes = {}

/** user starts loading url */
function start(URL) {
    startTimes[URL] = new Date();
    // td: load sth bg? ;-)
    console.log('do something periodically');
}
exports.start = URL => start(URL);

/** user ends loading url */
function stop(URL) {
    delete startTimes[URL];
    console.log('end do');
}
exports.stop = URL => stop(URL);

/** currently loading? */
function isIdle() {
    return _.isEmpty(startTimes);
}
exports.isIdle = () => isIdle();
    
