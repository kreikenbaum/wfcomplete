"use strict";

exports.DOC = 'creates cover traffic, up to predetermined parameters';

const coverUrl = require("./coverUrl.js");
const debug = require("./debug.js");
const stats = require("./stats.js");

// td: changeable by pref (slider?)
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1.5;

function CoverTraffic(load) {
    this.load = load;

    this.site = {};
    this.pad = {};

    this.site.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    this.pad.html = stats.htmlSize(FACTOR) - this.site.html;
    this.site.numEmbedded = stats.numberEmbeddedObjects(1); // td: see above
    this.pad.numEmbedded
	= stats.numberEmbeddedObjects(FACTOR) - this.site.numEmbedded;

    this.prob = Math.max(this.pad.numEmbedded / this.site.numEmbedded, 0);

    load.http(coverUrl.sized(this.pad.html));
}
// td: 
CoverTraffic.prototype.loadNext = function() {
    var i = (stats.withProbability( this.prob % 1 )
	     ? Math.ceil(this.prob)
	     : Math.floor(this.prob));
    for ( ; i >= 0; i -= 1 ) {
	if ( i > 1 || stats.withProbability(i) ) {
	    load.http(coverUrl.sized(stats.embeddedObjectSize()));
	}
    }
};
exports.CoverTraffic = CoverTraffic;

var load;

var site_ = {};
var pad_ = {};
var prob_;

function setLoader(load_param) {
    load = load_param;
}
exports.setLoader = (load_param) => setLoader(load_param);

/** a website is loaded by the user. covertraffic determines a target
 * feature vector and adds to the load to approximate the target.
 */
function start() {
    site_ = {};
    pad_ = {};

    site_.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    pad_.html = stats.htmlSize(FACTOR) - site_.html;
    site_.numEmbedded = stats.numberEmbeddedObjects(1); // td: see above
    pad_.numEmbedded = stats.numberEmbeddedObjects(FACTOR) - site_.numEmbedded;
    prob_ = Math.max(pad_.numEmbedded / site_.numEmbedded, 0);

    //    setTimeout(loadNext, stats.parsingTime());
    load.http(coverUrl.sized(pad_.html));
    debug.log("site_: " + JSON.stringify(site_));
    debug.log("pad_: " + JSON.stringify(pad_));
}
exports.start = start;

// td: timeout
function loadNext() {
    var i;
    if ( pad_.numEmbedded > 0 ) {
	for ( i = prob_; i >= 0; i -= 1 ) {
	    if ( i > 1 || stats.withProbability(i) ) {
		load.http(coverUrl.sized(stats.embeddedObjectSize()));
		pad_.numEmbedded -= 1;
	    }
	}
    } else {
	console.log("cover traffic empty, num: " + pad_.numEmbedded);
    }
}
exports.loadNext = loadNext;
