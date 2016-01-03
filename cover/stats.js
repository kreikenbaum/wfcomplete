/**
* statistical models for distributions etc
*/
const htmlMu = 7.90272;
const htmlSigma = 1.7643;

var htmlSize = function(factor = 2) {
    return factor * lognormal(htmlMu, htmlSigma);
};
exports.htmlSize = factor => htmlSize(factor);

// for testing ;-)
/** mean of lognormal(htmlMu, htmlSigma) */
var htmlMean = function() {
    return Math.exp(htmlMu + htmlSigma * htmlSigma / 2);
};
exports.htmlMean = htmlMean;

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

// td: make crypto-secure
/** sample from uniform distribution*/
function uniform(min, max) {
    return Math.random() * (max - min) + min;
};
