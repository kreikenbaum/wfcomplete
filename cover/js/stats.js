"use strict";

exports.DOC = 'distributions, also application of distributions (html etc)';

const random = require("js/random.js");

const HTML_MU = 7.90272;
const HTML_SIGMA = 1.7643;
const EMBEDDED_SIZE_MU = 7.51384;
const EMBEDDED_SIZE_SIGMA = 2.17454;
const EMBEDDED_NUM_KAPPA = 0.141385;
const EMBEDDED_NUM_THETA = 40.3257;
const PARSINGTIME_MU = -1.24892;
const PARSINGTIME_SIGMA = 2.08427;
const REQUEST_LENGTH_MAX = 700;

// td: refactor, remove "factor"
/** @return expected size of html response * factor */
function htmlSize(factor = 1) {
    return factor * lognormal_(HTML_MU, HTML_SIGMA);
};
exports.htmlSize = (factor) => htmlSize(factor);

/** @return expected size of embedded object response * factor */
function embeddedObjectSize(factor = 1) {
    return factor * lognormal_(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA);
}
exports.embeddedObjectSize = (factor) => embeddedObjectSize(factor);

/** returns at least 1, at most ceil(factor * numEmbedded) */
function numberEmbeddedObjects(factor=1) {
    return Math.max(1, Math.ceil(factor * gamma_(EMBEDDED_NUM_KAPPA,
						 EMBEDDED_NUM_THETA)));
}
exports.numberEmbeddedObjects = (factor) => numberEmbeddedObjects(factor);

// // td: test if used
// function parsingTime() {
//     return lognormal_(PARSINGTIME_MU, PARSINGTIME_SIGMA);
// }
// exports.parsingTime = parsingTime;

function requestLength() {
    return uniform_(0, REQUEST_LENGTH_MAX);
}
exports.requestLength = requestLength;

/** returns true with a probability of <code>chance</code> (for chance
 * >1: always)*/
function withProbability(chance) {
    return random.uniform01() < chance;
}
exports.withProbability = withProbability;

// TESTING
/** mean of lognormal_(HTML_MU, HTML_SIGMA) */
function htmlMean() {
    return Math.exp(HTML_MU + HTML_SIGMA * HTML_SIGMA / 2);
}
exports.htmlMean = htmlMean;
/** mean of lognormal_(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA) */
function embeddedObjectMean() {
    return Math.exp(EMBEDDED_SIZE_MU + EMBEDDED_SIZE_SIGMA * EMBEDDED_SIZE_SIGMA / 2);
}
exports.embeddedObjectMean = embeddedObjectMean;



/** sample from lognormal_ distribution */
function lognormal_(mu, sigma) {
    return Math.exp(normal_(mu, sigma));
}

/** sample from normal distribution */
function normal_(mu, sigma) {
    var p1, p2;
    var p = 2;
    while ( p > 1.0 ) {
	p1 = uniform_(-1, 1);
	p2 = uniform_(-1, 1);
        p = p1 * p1 + p2 * p2;
    }
    return mu + sigma * p1 * Math.sqrt(-2 * Math.log(p) / p);
}

/** sample from uniform distribution*/
function uniform_(min, max) {
    return random.uniform01() * (max - min) + min;
}

// code courtesy of "Computer Generation of Statistical Distributions"
// sec 5.1.11 (with a = 0, b = theta, c = kappa)
/** sample from gamma distribution, code valid for kappa < 1 */
function gamma_(kappa, theta) {
    const C = 1 + kappa / Math.E;
    while ( true ) {
	var p = C * random.uniform01();
	if ( p > 1 ) {
	    var y = - Math.log( (C - p ) / kappa );
	    if ( random.uniform01() <= Math.pow(y, kappa-1) ) {
		return 0;
	    }
	} else {
	    var y = Math.pow(p, 1/kappa);
	    if ( random.uniform01() <= Math.exp(-y) ) {
		return theta * y;
	    }
	}
    }
}