"use strict";

exports.DOC = 'creates cover traffic, up to predetermined parameters';

const { setTimeout } = require("sdk/timers");

const coverUrl = require("./coverUrl.js");
const stats = require("./stats.js");

// td: changeable by pref (slider?)
/** overhead of dummy traffic -1; 1.5 has overhead of 50% */
const FACTOR = 1.5;

var load;

var site_ = {};
var pad_ = {};
var prob_;

function setLoader(load_param) {
    load = load_param;
}
exports.setLoader = load_param => setLoader(load_param);

/** a website is loaded by the user. covertraffic determines a target
 * feature vector and adds to the load to approximate the target.
 */
function start() {
    site_ = {};
    pad_ = {};

    site_.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    pad_.html = stats.htmlSize(FACTOR) - site_.html;
    site_.num_embedded = stats.numberEmbeddedObjects(1); // td: see above
    pad_.num_embedded = stats.numberEmbeddedObjects(FACTOR) - site_.num_embedded;
    prob_ = pad_.num_embedded / site_.num_embedded;

    //    setTimeout(loadNext, stats.parsingTime());
    load.http(coverUrl.sized(pad_.html));
};
exports.start = start;

// td: timeout
function loadNext() {
    var i;
    if ( pad_.num_embedded > 0 ) {
	for ( i = prob_; i >= 0; i -= 1 ) {
	    if ( i > 1 || stats.withProbability(i) ) {
		load.http(coverUrl.sized(stats.embeddedObjectSize()));
		pad_.num_embedded -= 1;
	    }
	}
    } else {
	console.log("cover traffic empty, num: " + pad_.num_embedded);
    }
};
exports.loadNext = loadNext;
