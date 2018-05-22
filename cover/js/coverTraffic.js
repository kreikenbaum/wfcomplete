"use strict";

/** @fileoverview Determines a target feature vector and adds to the
 * load to approximate the target. The HTML call is done on creation,
 * embedded object calls on {@code loadNext}.
 */
const Simple = require('sdk/simple-prefs');

const sizeCache = require("./size-cache.js");
const stats = require("./stats.js");
const LOAD = require('./proxy_sum_load.js');

/** overhead of dummy traffic; 1.5 created by *factor* 50 */
const FACTOR = 1 + (Simple.prefs.factor / 100);

/**
 * Creates cover traffic. Once on creation, then on {@code loadNext()}.
 * @constructor
 * @param {module load.js} module which provides loading capabilities
 * @param {String targetURLSpec} first request to site was this URL
 */
function CoverTraffic(targetURLSpec, load=LOAD) {
    this.load = load;

    this.site = {};
    this.target = {};
    this.url = targetURLSpec; // testing
    this.redirects = 0;

    if ( Simple.prefs.sizes ) { //  A: KNOWN SIZES IF KNOWN
        this.site.html = this.htmlStrategyKnown(targetURLSpec);
        this.site.numEmbedded = this.numStrategyKnown(targetURLSpec);
    } else {                    //  B: ONLY GUESSED SIZES
        this.site.html = this.htmlStrategyGuess(targetURLSpec);
        this.site.numEmbedded = this.numStrategyGuess(targetURLSpec);
    }

    if ( Simple.prefs.bins ) { //  I: BLOOM BIN MAX
        this.target.html = this.htmlStrategyBins(targetURLSpec);
        this.target.numEmbedded = this.numStrategyBins(targetURLSpec);
    } else {                   //  II: ONE TARGET DISTRIBUTION
        this.target.html = this.htmlStrategyDist(targetURLSpec);
        this.target.numEmbedded = this.numStrategyDist(targetURLSpec);
    }

    this.prob = Math.max(
        minProb_(this.site.numEmbedded),
        Math.min(this.target.numEmbedded / this.site.numEmbedded,
             2 * Simple.prefs.factor));
    this.target.origNumEmbedded = this.target.numEmbedded;

    //    console.log(this);
    this.load.sized(this.target.html);
}

// disabled by default for version 0.18+
CoverTraffic.prototype.finish = function() {
    console.log('finish() traffic to ' + this.url +
                ' with ' + this.target.numEmbedded + ' to load');
    if ( Simple.prefs.burst ) {
        while ( this.target.numEmbedded > 0 ) {
            this.load.sized(stats.embeddedObjectSize());
            this.target.numEmbedded -= 1;
        }
    }
};

// td: think about bursts at end, maybe browser connection-per-site delay is
// enough to create multiple bursts
CoverTraffic.prototype.loadNext = function() {
  // todo: maybe just remove the redirect stuff, test if still good
    if ( this.redirects > 0 && stats.withProbability(Simple.prefs.redirect_p /100) ) {
        this.load.sized(Simple.prefs.min_size);
        this.redirects -= 1;
    }

    // do once for each 1-integer part and maybe for the fraction
    for ( let i = (stats.withProbability( this.prob % 1 ) ? Math.ceil(this.prob)
                   : Math.floor(this.prob)) ;
          i >= 1 ;
          i -= 1 ) {
        if ( this.target.numEmbedded > 0 ) {
            this.load.sized(stats.embeddedObjectSize());
            this.target.numEmbedded -= 1;
        } else {
            console.log('reached numEmbedded: ' + JSON.stringify(this));
        }
    }
};

/** previous traffic got a HTTP-redirect status */
CoverTraffic.prototype.redirected = function() {
    this.redirects += 1;
};

// ### strategies I/II: bloom bin max or one target distribution
// td later: strategy class, subclasses
/** partial Strategy I: take bloom max for target */
// last bin handled by BloomSort
CoverTraffic.prototype.htmlStrategyBins = function(targetURLSpec) {
    let targetHtmlSize;
    try {
        targetHtmlSize = sizeCache.htmlSizeMax(targetURLSpec);
    } catch (e) {
        targetHtmlSize = stats.htmlSize(FACTOR);
    }
    return targetHtmlSize - this.site.html;
};
// td: is this code duplication
CoverTraffic.prototype.numStrategyBins = function(targetURLSpec) {
    let targetPadSize;
    try {
        targetPadSize = sizeCache.numberEmbeddedObjectsMax(targetURLSpec);
    } catch (e) {
        targetPadSize = stats.numberEmbeddedObjects(FACTOR);
    }
    return targetPadSize - this.site.numEmbedded;
};

/** partial Strategy II: use one distribution as target */
CoverTraffic.prototype.htmlStrategyDist = function(targetURLSpec) {
    return stats.htmlSize(FACTOR) - this.site.html;
};
CoverTraffic.prototype.numStrategyDist = function(targetURLSpec) {
    return stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;
};

// ### strategies A/B know size or guess size
/** partial Strategy A: @return known sizes for sites */
CoverTraffic.prototype.htmlStrategyKnown = function(targetURLSpec) {
    try {
        return sizeCache.htmlSize(targetURLSpec);
        //console.log('site size hit for ' + targetURLSpec + ": " + this.site.html);
    } catch (e) { // guess sizes
        return stats.htmlSize();
        //console.log('site size miss for ' + targetURLSpec + ": " +JSON.stringify(e));
    }
};
CoverTraffic.prototype.numStrategyKnown = function(targetURLSpec) {
    try {
        return sizeCache.numberEmbeddedObjects(targetURLSpec);
    } catch (e) { // guess sizes
        return stats.numberEmbeddedObjects();
    }
};
/** partial Strategy B: @return random sizes for sites */
CoverTraffic.prototype.htmlStrategyGuess = function(targetURLSpec) {
    return stats.htmlSize();
};
CoverTraffic.prototype.numStrategyGuess = function(targetURLSpec) {
    return stats.numberEmbeddedObjects();
};

exports.CoverTraffic = CoverTraffic;

/** @return minimum probability of embedded retrieval:

Computes as sqrt(n) / n where n is the site's number of embedded
objects */
function minProb_(numEmbeddedSite) {
    return Math.sqrt(numEmbeddedSite) / numEmbeddedSite;
}

