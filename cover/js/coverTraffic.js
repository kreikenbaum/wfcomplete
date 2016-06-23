"use strict";

/** @fileoverview Determines a target feature vector and adds to the
 * load to approximate the target. The HTML call is done on creation,
 * embedded object calls on {@code loadNext}.
 */
const Simple = require('sdk/simple-prefs');

const debug = require("./debug.js");
const sizeCache = require("./size-cache.js");
const stats = require("./stats.js");

// prefs allow only integers, no floats
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1 + (Simple.prefs.factor / 100);
/** minimum probability of also requesting a cover element per embedded */

//const LOAD = require('./load.js');
const LOAD = require('./proxy_sum_load.js');

/**
 * Creates cover traffic. Once on creation, then on {@code loadNext()}.
 * @constructor
 * @param {module load.js} module which provides loading capabilities
 * @param {String targetURL} first request to site was this URL
 */
function CoverTraffic(targetURL, load=LOAD) {
    this.load = load;

    this.site = {};
    this.target = {};
    this.url = targetURL; // testing

    if ( Simple.prefs.sizes ) { //  A: KNOWN SIZES IF KNOWN
	this.site.html = this.htmlStrategyKnown(targetURL);
	this.site.numEmbedded = this.numStrategyKnown(targetURL);
    } else {                    //  B: ONLY GUESSED SIZES
	this.site.html = this.htmlStrategyGuess(targetURL);
	this.site.numEmbedded = this.numStrategyGuess(targetURL);
    }

    if ( Simple.prefs.bins ) { //  I: BLOOM BIN MAX
	this.target.html = this.htmlStrategyBins(targetURL);
	this.target.numEmbedded = this.numStrategyBins(targetURL);
    } else {                   //  II: ONE TARGET DISTRIBUTION
	this.target.html = this.htmlStrategyDist(targetURL);
	this.target.numEmbedded = this.numStrategyDist(targetURL);
    }

    // try: min <= ratio <= average_request_num (*extra* requests)
    this.prob = Math.max(minProb_(this.site.numEmbedded),
                         Math.min(
                             this.target.numEmbedded / this.site.numEmbedded,
                             stats.numberEmbeddedObjectsMean()));
    //    console.log(this);

    this.load.sized(this.target.html);
}

// td: think about this, maybe browser connection-per-site delay is
// enough to create multiple bursts
CoverTraffic.prototype.loadNext = function() {
    // do once for each 1-integer part and maybe for the fraction
    for ( let i = (stats.withProbability( this.prob % 1 ) ? Math.ceil(this.prob)
		   : Math.floor(this.prob)) ;
	  i >= 0 ;
	  i -= 1 ) {
// v21        if ( this.target.numEmbedded > 0 ) {
	this.load.sized(stats.embeddedObjectSize());
        this.target.numEmbedded -= 1;
// end v21        }
    }
};


// disabled by default for version 0.18+
// maybe balanced out with too many requests (target.numEmbedded <0 not checked)
CoverTraffic.prototype.finish = function() {
    debug.log('ending traffic with ' + this.target.numEmbedded + ' to load');
    if ( Simple.prefs.burst ) {
	while ( this.target.numEmbedded > 0 ) {
	    this.load.sized(stats.embeddedObjectSize());
	    this.target.numEmbedded -= 1;
	}
    }
};

// ### strategies I/II: bloom bin max or one target distribution
// td later: strategy class, subclasses
/** partial Strategy I: take bloom max for target */
// last bin handled by BloomSort
CoverTraffic.prototype.htmlStrategyBins = function(targetURL) {
    let targetHtmlSize;
    try {
	targetHtmlSize = sizeCache.htmlSizeMax(targetURL);
    } catch (e) {
	targetHtmlSize = stats.htmlSize(FACTOR);
    }
    return targetHtmlSize - this.site.html;
};
// td: is this code duplication
CoverTraffic.prototype.numStrategyBins = function(targetURL) {
    let targetPadSize;
    try {
	targetPadSize = sizeCache.numberEmbeddedObjectsMax(targetURL);
    } catch (e) {
	targetPadSize = stats.numberEmbeddedObjects(FACTOR);
    }
    return targetPadSize - this.site.numEmbedded;
};

/** partial Strategy II: use one distribution as target */
CoverTraffic.prototype.htmlStrategyDist = function(targetURL) {
    return stats.htmlSize(FACTOR) - this.site.html;
};
CoverTraffic.prototype.numStrategyDist = function(targetURL) {
    return stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;
};

// ### strategies A/B know size or guess size
/** partial Strategy A: @return known sizes for sites */
CoverTraffic.prototype.htmlStrategyKnown = function(targetURL) {
    try {
	return sizeCache.htmlSize(targetURL);
	//console.log('site size hit for ' + targetURL + ": " + this.site.html);
    } catch (e) { // guess sizes
	return stats.htmlSize();
	//console.log('site size miss for ' + targetURL + ": " +JSON.stringify(e));
    }
};
CoverTraffic.prototype.numStrategyKnown = function(targetURL) {
    try {
	return sizeCache.numberEmbeddedObjects(targetURL);
    } catch (e) { // guess sizes
	return stats.numberEmbeddedObjects();
    }
};
/** partial Strategy B: @return random sizes for sites */
CoverTraffic.prototype.htmlStrategyGuess = function(targetURL) {
    return stats.htmlSize();
};
CoverTraffic.prototype.numStrategyGuess = function(targetURL) {
    return stats.numberEmbeddedObjects();
};

exports.CoverTraffic = CoverTraffic;

/** @return minimum probability of embedded retrieval:

Computes as sqrt(n) / n where n is the site's number of embedded
objects */
function minProb_(numEmbeddedSite) {
    return Math.sqrt(numEmbeddedSite) / numEmbeddedSite;
}

