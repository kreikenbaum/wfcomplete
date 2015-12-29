/** debug output */
const DEBUG = true;

/** use as drop-in replacement to console.log (which logs as xpi even though it should not */
function log(toLog) {
    if ( DEBUG ) { console.log(toLog); }
}
exports.log = toLog => log(toLog);
