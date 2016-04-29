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
 */
function CoverTraffic(targetURL, load=LOAD) {
    this.load = load;

    this.site = {};
    this.target = {};

    try {
	this.site.html = sizeCache.htmlSize(targetURL);
	//console.log('site size hit for ' + targetURL + ": " + this.site.html);
    } catch (e) { // guess sizes
	this.site.html = stats.htmlSize();
	//console.log('site size miss for ' + targetURL + ": " +JSON.stringify(e));
    }
    try {
	this.site.numEmbedded = sizeCache.numberEmbeddedObjects(targetURL);
    } catch (e) { // guess sizes
	this.site.numEmbedded = stats.numberEmbeddedObjects();
    }

    //    this.target.html = stats.htmlSize(FACTOR) - this.site.html;
    this.target.html = this.htmlStrategy1(targetURL);
    this.target.numEmbedded = this.numStrategy1(targetURL);
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
 * Strategy 1: take bloom max
 * on bloom failure: (v0.1) take default * FACTOR
 */
// td: better handling of last bucket
CoverTraffic.prototype.htmlStrategy1 = function(targetURL) {
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
CoverTraffic.prototype.numStrategy1 = function(targetURL) {
    let targetPadSize;
    try {
	targetPadSize = sizeCache.numberEmbeddedObjectsMax(targetURL);
    } catch (e) {
	targetPadSize = stats.numberEmbeddedObjects(FACTOR);
    }
    return targetPadSize - this.site.numEmbedded;
};

exports.CoverTraffic = CoverTraffic;

	// td: evaluate
	// if ( e.message === 'The last bucket has no border' ) {
	//     return stats.HTML_TOP;
	//     return stats.HTML_999;
	//     return HTML_SIZES[HTML_SIZES.length -1];
	//     return HTML_SIZES[HTML_SIZES.length -1] * FACTOR; // favorite
	// } else {
//	    return stats.htmlSize();
	// }
