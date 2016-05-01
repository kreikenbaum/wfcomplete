"use strict";

/** @fileoverview Determines a target feature vector and adds to the
 * load to approximate the target. The HTML call is done on creation,
 * embedded object calls on {@code loadNext}.
 */
const Simple = require('sdk/simple-prefs');

const debug = require("./debug.js");
const sizeCache = require("./size-cache.js");
const stats = require("./stats.js");

// td: auto-update pref
// prefs allow only integers, no floats
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1 + (Simple.prefs.factor / 100);
/** minimum probability of also requesting a cover element per embedded */
const MIN_PROB = 0.1; // td: test and think through

const LOAD = require('./load.js');

/**
 * Creates cover traffic. Once on creation, then on {@code loadNext()}.
 * @constructor
 * @param {module load.js} module which provides loading capabilities
 * @param {String targetURL} first request to site was this URL
 */
// td: refactor: member variable for targetURL?
function CoverTraffic(targetURL, load=LOAD) {
    this.load = load;

    this.site = {};
    this.target = {};

//    this.site.html = this.htmlStrategyA(targetURL);
//    this.site.numEmbedded = this.numStrategyA(targetURL);
    this.site.html = this.htmlStrategyB(targetURL);
    this.site.numEmbedded = this.numStrategyB(targetURL);

    //    this.target.html = stats.htmlSize(FACTOR) - this.site.html;
    this.target.html = this.htmlStrategyI(targetURL);
    this.target.numEmbedded = this.numStrategyI(targetURL);
//	stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;

    this.prob = Math.max(this.target.numEmbedded / this.site.numEmbedded,
			 MIN_PROB); // td: maybe * Math.sqrt(site.numEmbedded)

    this.load.sized(this.target.html);
}

CoverTraffic.prototype.loadNext = function() {
    var i;
    // do once for each 1-integer part and maybe for the fraction
    for ( i = (stats.withProbability( this.prob % 1 ) ? Math.ceil(this.prob)
	       : Math.floor(this.prob)) ;
	  i >= 0 ;
	  i -= 1 ) {
	this.load.sized(stats.embeddedObjectSize());
        this.target.numEmbedded -= 1;
    }
};

// td later: strategy class, subclasses
/**
 * partial Strategy I: take bloom max for target
 * on bloom failure: (v0.1) take default * FACTOR
 */
// td: better handling of last bucket
CoverTraffic.prototype.htmlStrategyI = function(targetURL) {
    let targetHtmlSize;
    try {
	targetHtmlSize = sizeCache.htmlSizeMax(targetURL);
	//console.log('target size hit for ' + targetURL + ": " + targetHtmlSize);
    } catch (e) {
	targetHtmlSize = stats.htmlSize(FACTOR);
	//console.log('size miss for ' + JSON.stringify(targetURL)
	//    + 'due to: ' + JSON.stringify(e));
    }
    return targetHtmlSize - this.site.html;
};
// td: is this code duplication
CoverTraffic.prototype.numStrategyI = function(targetURL) {
    let targetPadSize;
    try {
	targetPadSize = sizeCache.numberEmbeddedObjectsMax(targetURL);
    } catch (e) {
	targetPadSize = stats.numberEmbeddedObjects(FACTOR);
    }
    return targetPadSize - this.site.numEmbedded;
};

/**
* partial Strategy II: one distribution for target
*/
// td

/**
* partial Strategy A: known sizes for sites
*/
CoverTraffic.prototype.htmlStrategyA = function(targetURL) {
    try {
	return sizeCache.htmlSize(targetURL);
	//console.log('site size hit for ' + targetURL + ": " + this.site.html);
    } catch (e) { // guess sizes
	return stats.htmlSize();
	//console.log('site size miss for ' + targetURL + ": " +JSON.stringify(e));
    }
};
CoverTraffic.prototype.numStrategyA = function(targetURL) {
    try {
	return sizeCache.numberEmbeddedObjects(targetURL);
    } catch (e) { // guess sizes
	return stats.numberEmbeddedObjects();
    }
};
/**
* partial Strategy B: random sizes for sites
*/
CoverTraffic.prototype.htmlStrategyB = function(targetURL) {
    return stats.htmlSize();
};
CoverTraffic.prototype.numStrategyB = function(targetURL) {
    return stats.numberEmbeddedObjects();
};

exports.CoverTraffic = CoverTraffic;
