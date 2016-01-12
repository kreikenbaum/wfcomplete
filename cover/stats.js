"use strict";

exports.DOC = 'statistical models for distributions etc';

const random = require("./random.js");

const HTML_MU = 7.90272;
const HTML_SIGMA = 1.7643;
const EMBEDDED_SIZE_MU = 7.51384;
const EMBEDDED_SIZE_SIGMA = 7.51384;
const EMBEDDED_NUM_KAPPA = 0.141385;
const EMBEDDED_NUM_THETA = 40.3257;

// td: refactor, remove "factor"
/** @return expected size of html response * factor */
var htmlSize = function(factor = 2) {
    return factor * lognormal(HTML_MU, HTML_SIGMA);
};
exports.htmlSize = factor => htmlSize(factor);

/** @return expected size of embedded object response * factor */
var embeddedObjectSize = function(factor = 2) {
    return factor * lognormal(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA);
}
exports.embeddedObjectSize = factor => embeddedObjectSize(factor);

var numberEmbeddedObjects = function(factor = 2) {
    return factor * gamma(EMBEDDED_NUM_KAPPA, EMBEDDED_NUM_THETA);
}
exports.numberEmbeddedObjects = factor => numberEmbeddedObjects(factor);

// TESTING
/** mean of lognormal(HTML_MU, HTML_SIGMA) */
var htmlMean = function() {
    return Math.exp(HTML_MU + HTML_SIGMA * HTML_SIGMA / 2);
};
exports.htmlMean = htmlMean;
/** mean of lognormal(EMBEDDED_SIZE_MU, EMBEDDED_SIZE_SIGMA) */
var embeddedObjectMean = function() {
    return Math.exp(EMBEDDED_SIZE_MU + EMBEDDED_SIZE_SIGMA * EMBEDDED_SIZE_SIGMA / 2);
};
exports.embeddedObjectMean = embeddedObjectMean;

// PRIVATE
/** sample from lognormal distribution */
function lognormal(mu, sigma) {
    return Math.exp(normal(mu, sigma));
};

/** sample from normal distribution */
function normal(mu, sigma) {
    var p1, p2;
    var p = 2;
    while ( p > 1.0 ) {
	p1 = uniform(-1, 1);
	p2 = uniform(-1, 1);
        p = p1 * p1 + p2 * p2;
    }
    return mu + sigma * p1 * Math.sqrt(-2. * Math.log(p) / p);
};

/** sample from uniform distribution*/
function uniform(min, max) {
    return random.uniform01() * (max - min) + min;
};

// code courtesy of "Computer Generation of Statistical Distributions"
// sec 5.1.11 (with a = 0, b = theta, c = kappa)
/** sample from gamma distribution, code valid for kappa < 1 */
function gamma(kappa, theta) {
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
};
