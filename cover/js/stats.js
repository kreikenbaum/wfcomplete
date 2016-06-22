"use strict";
/**
* @fileoverview Some basic distributions with their HTTP-related
* applications for modelling web traffic.
*/
const random = require("js/random.js");

const HTML_MU = 7.90272;
const HTML_SIGMA = 1.7643;
const HTML_TOP = 2 * 1024 * 1024; // 2 MB
exports.HTML_TOP = HTML_TOP; // testing

const EMBEDDED_SIZE_MU = 7.51384;
const EMBEDDED_SIZE_SIGMA = 2.17454;
const EMBEDDED_SIZE_TOP = 6 * 1024 * 1024; // 6 MB
exports.EMBEDDED_SIZE_TOP = EMBEDDED_SIZE_TOP; // testing

const EMBEDDED_NUM_KAPPA = 0.141385;
const EMBEDDED_NUM_THETA = 40.3257;
const EMBEDDED_NUM_TOP = 300;
exports.EMBEDDED_NUM_TOP = EMBEDDED_NUM_TOP; // testing

//const PARSINGTIME_MU = -1.24892;
//const PARSINGTIME_SIGMA = 2.08427;
const REQUEST_LENGTH_MAX = 700;

// td: refactor, remove "factor"
/** @return (expected size of html response) * factor */
function htmlSize(factor=1) {
    let out = lognormal_(HTML_MU, HTML_SIGMA);
    while ( out > HTML_TOP ) {
        out = lognormal_(HTML_MU, HTML_SIGMA);
    }
    return factor * out;
}
exports.htmlSize = (factor) => htmlSize(factor);

/** @return expected size of embedded object response * factor */
function embeddedObjectSize(factor = 1) {
    let out = lognormal_(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA);
    while ( out > EMBEDDED_SIZE_TOP ) {
	out = lognormal_(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA);
    }
    return factor * out;
}
exports.embeddedObjectSize = (factor) => embeddedObjectSize(factor);

/** returns at least 1, at most ceil(factor * numEmbedded) */
function numberEmbeddedObjects(factor=1) {
    let out = Math.max(1, Math.ceil(factor * gamma_(EMBEDDED_NUM_KAPPA,
						    EMBEDDED_NUM_THETA)));
    while ( out > EMBEDDED_SIZE_TOP ) {
	out = Math.max(1, Math.ceil(factor * gamma_(EMBEDDED_NUM_KAPPA,
						    EMBEDDED_NUM_THETA)));
    }
    return factor * out;
}
exports.numberEmbeddedObjects = (factor) => numberEmbeddedObjects(factor);

// // td: test if used
// function parsingTime() {
//     return lognormal_(PARSINGTIME_MU, PARSINGTIME_SIGMA);
// }
// exports.parsingTime = parsingTime;

function requestLength() {
    return uniform(0, REQUEST_LENGTH_MAX);
}
exports.requestLength = requestLength;

/** @return true with a probability of <code>chance</code> (for chance
 * >1: always)*/
function withProbability(chance) {
    return random.uniform01() < chance;
}
exports.withProbability = withProbability;

/** sample from uniform distribution*/
function uniform(min=0, max=1) {
    return random.uniform01() * (max - min) + min;
}
exports.uniform = uniform;

// TESTING
/** mean of lognormal_(HTML_MU, HTML_SIGMA) */
function htmlMean() {
//    return Math.exp(HTML_MU + HTML_SIGMA * HTML_SIGMA / 2); // non-truncated
    return 11872; // truncated at 2MB, value from paper
}
exports.htmlMean = htmlMean;
/** mean of lognormal_(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA) */
function embeddedObjectSizeMean() {
//    return Math.exp(EMBEDDED_SIZE_MU + EMBEDDED_SIZE_SIGMA * EMBEDDED_SIZE_SIGMA / 2);
    return 12460; // truncated at 6MB, value from paper
}
exports.embeddedObjectSizeMean = () => embeddedObjectSizeMean();

function embeddedObjectSizeStd() {
    return 116050; // value from paper
}
exports.embeddedObjectSizeStd = () => embeddedObjectSizeStd();

function numberEmbeddedObjectsMean() {
    return EMBEDDED_NUM_KAPPA * EMBEDDED_NUM_THETA;
}
exports.numberEmbeddedObjectsMean = () => numberEmbeddedObjectsMean();

function numberEmbeddedObjectsVar() {
    return EMBEDDED_NUM_KAPPA * EMBEDDED_NUM_THETA * EMBEDDED_NUM_THETA;
}
exports.numberEmbeddedObjectsVar = () => numberEmbeddedObjectsVar();

// PRIVATE
/** sample from lognormal_ distribution */
function lognormal_(mu, sigma) {
    return Math.exp(normal_(mu, sigma));
}

/** sample from normal distribution */
function normal_(mu, sigma) {
    var p1, p2;
    var p = 2;
    while ( p > 1.0 ) {
	p1 = uniform(-1, 1);
	p2 = uniform(-1, 1);
        p = p1 * p1 + p2 * p2;
    }
    return mu + sigma * p1 * Math.sqrt(-2 * Math.log(p) / p);
}

// code courtesy of "Computer Generation of Statistical Distributions"
// sec 5.1.11 (with a = 0, b = theta, c = kappa)
/** sample from gamma distribution, code valid for kappa < 1 */
function gamma_(kappa, theta) {
    const C = 1 + kappa / Math.E;
    while ( true ) {
	var p = C * random.uniform01();
	var y;
	if ( p > 1 ) {
	    y = - Math.log( (C - p ) / kappa );
	    if ( random.uniform01() <= Math.pow(y, kappa-1) ) {
		return 0;
	    }
	} else {
	    y = Math.pow(p, 1/kappa);
	    if ( random.uniform01() <= Math.exp(-y) ) {
		return theta * y;
	    }
	}
    }
}
