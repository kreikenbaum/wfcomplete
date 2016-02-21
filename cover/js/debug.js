/** debug output */
"use strict";

const DEBUG = true;
const TRACE = false;

/** use as drop-in replacement to console.log (which logs as xpi even though it should not */
function log(toLog) {
    if ( DEBUG ) {
	console.log(toLog);
    }
}
exports.log = (toLog) => log(toLog);

/** for really verbose output, enable only when needed */
function traceLog(toLog) {
    if ( TRACE ) {
	console.log(toLog);
    }
}
exports.traceLog = (toLog) => traceLog(toLog);
