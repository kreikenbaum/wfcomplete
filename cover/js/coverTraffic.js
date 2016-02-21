"use strict";

/** @fileoverview Determines a target feature vector and adds to the
 * load to approximate the target. The HTML call is done on creation,
 * embedded object calls on {@code loadNext}.
 */
exports.DOC = 'creates cover traffic, up to predetermined parameters';

const coverUrl = require("./coverUrl.js");
const debug = require("./debug.js");
const stats = require("./stats.js");

// td: changeable by pref (slider?)
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1.5;
const MIN_PROB = 0.1; // td: test and think through

function CoverTraffic(load) {
    this.load = load;

    this.site = {};
    this.pad = {};

    this.site.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    this.pad.html = stats.htmlSize(FACTOR) - this.site.html;
    this.site.numEmbedded = stats.numberEmbeddedObjects(1); // td: see above
    this.pad.numEmbedded =
	stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;

    this.prob = Math.max(this.pad.numEmbedded / this.site.numEmbedded, MIN_PROB);

    this.load.http(coverUrl.sized(this.pad.html));
}

CoverTraffic.prototype.loadNext = function() {
    var i;
    for ( i = (stats.withProbability( this.prob % 1 ) ? Math.ceil(this.prob)
	       : Math.floor(this.prob)) ;
	  i >= 0 ;
	  i -= 1 ) {
	this.load.http(coverUrl.sized(stats.embeddedObjectSize()));
    }
};
exports.CoverTraffic = CoverTraffic;
