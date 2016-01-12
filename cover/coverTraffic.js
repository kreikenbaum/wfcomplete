"use strict";

exports.DOC = 'creates cover traffic, up to predetermined parameters';

const stats = require("./stats.js");

const FACTOR = 1.5;

/** a website is loaded by the user. covertraffic determines a target
 * feature vector and adds to the load to approximate the target
 */
function CoverTraffic() {
    var site = {};
    var pad = {};
    site.html = stats.htmlSize(1); // td: buffer known sites in bloomfilter
    pad.html = stats.htmlSize(FACTOR) - site.html;
    site.num_embedded = stats.numberEmbeddedObjects(1);
    pad.num_embedded = stats.numberEmbeddedObjects(FACTOR) - site.num_embedded;
    site.total_bytes = site.html;
    for ( var i = 0; i < site.num_embedded; i++ ) {
	site.total_bytes += stats.embeddedObjectSize(1);
    }
    pad.total_bytes = site.total_bytes * FACTOR;
    
