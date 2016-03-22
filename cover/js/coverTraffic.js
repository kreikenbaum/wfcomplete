"use strict";

/** @fileoverview Determines a target feature vector and adds to the
 * load to approximate the target. The HTML call is done on creation,
 * embedded object calls on {@code loadNext}.
 */
const Simple = require('sdk/simple-prefs');

const debug = require("./debug.js");
const sizeCache = require("./size-cache.js");
const stats = require("./stats.js");

// td: changeable by pref (slider?)
// prefs allow only integers, no floats
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1 + (Simple.prefs.factor / 100);
const MIN_PROB = 0.1; // td: test and think through

const LOAD = require('./load.js');

/**
 * Creates cover traffic. Once on creation, then on {@code loadNext()}.
 * @constructor
 * @param {module load.js} module which provides loading capabilities
 */
function CoverTraffic(toHost, load=LOAD) {
    this.load = load;

    this.site = {};
    this.pad = {};

    this.site.html = sizeCache.htmlSize(1);
    this.pad.html = stats.htmlSize(FACTOR) - this.site.html;
    this.site.numEmbedded = stats.numberEmbeddedObjects(1); // td: see above
    this.pad.numEmbedded =
	stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;

    this.prob = Math.max(this.pad.numEmbedded / this.site.numEmbedded,MIN_PROB);

    this.load.sized(this.pad.html);
}

CoverTraffic.prototype.loadNext = function() {
    var i;
    for ( i = (stats.withProbability( this.prob % 1 ) ? Math.ceil(this.prob)
	       : Math.floor(this.prob)) ;
	  i >= 0 ;
	  i -= 1 ) {
	this.load.sized(stats.embeddedObjectSize());
    }
};
exports.CoverTraffic = CoverTraffic;
