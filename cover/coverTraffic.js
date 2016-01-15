"use strict";

exports.DOC = 'creates cover traffic, up to predetermined parameters';

const { setTimeout } = require("sdk/timers");

const coverUrl = require("./coverUrl.js");
const load = require("./load.js");
const stats = require("./stats.js");

const FACTOR = 1.5;

var active = false;
var site_ = {};
var pad_ = {};

/** a website is loaded by the user. covertraffic determines a target
 * feature vector and adds to the load to approximate the target.
 */
function start() {
    // yes, this overrides previous activity
    active = true;
    site_ = {};
    pad_ = {};

    site_.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    pad_.html = stats.htmlSize(FACTOR) - site_.html;
    site_.num_embedded = stats.numberEmbeddedObjects(1); // td: see above
    pad_.num_embedded = stats.numberEmbeddedObjects(FACTOR) - site_.num_embedded;

    // td: extract links (and HEAD to get length)
    load.http(coverUrl.sized(pad_.html));
    
    setTimeout(loadNext, stats.parsingTime());
};
exports.start = start;

function loadNext() {
    if ( pad_.num_embedded > 0 && active === true ) {
	// td: determine range
	
	load.http(coverUrl.sized(stats.embeddedObjectSize()))
	
	pad_.num_embedded -= 1;
	setTimeout(loadNext, stats.parsingTime());
    } else {
	console.log("ending cover traffic, num: " + pad_.num_embedded);
    }
};
exports.loadNext = loadNext;

function stop() {
    active = false;
};
exports.stop = stop;
