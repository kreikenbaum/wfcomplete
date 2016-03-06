"use strict";
/**
* @fileoverview uses bloomfilters to sort elements
*/
const Bloom = require("../lib/bloomfilter.js");
const _ = require('../lib/underscore-min.js');

// td: typeerror if sizes.length !== splits.length + 1
// td: maybe test if sizes and splits are sorted and arrayed correctly
// td: generate splits if empty

/** 
 * @param {sizes Array} array of values for each bucket, must be sorted
 * @param {splits Array} array of bucket borders, must be sorted
*/
function BloomSort(sizes, splits) {
    this.sizes = sizes;
    this.num_filters = sizes.length;
    this.splits = splits; // || determine_splits(sizes);
    this.filters = [];
    for ( let i = 0; i < this.num_filters; i++ ) {
	this.filters[i] = new Bloom.BloomFilter(50, 3); // td:choose parameters 
    }
}
/** adds element of size {@code size} */
BloomSort.prototype.add = function(id, size) {
    this.filters[_.sortedIndex(this.splits, size)].add(id);
};
/** determines size of element, raises exception if unclear */
BloomSort.prototype.query = function(id) {
    let pos = -1;
    for ( let i = 0; i < this.filters.length; i++ ) {
	if ( this.filters[i].test(id) ) {
	    if ( pos === -1 ) {
		pos = i;
	    } else {
		throw {
		    name: 'BloomError',
		    message: 'Contains multiple entries'
		};
	    }
	}
    }
    if ( pos === -1 ) {
	throw {
	    name: 'BloomError',
	    message: 'Contains no entries'
	};
    }
    return this.sizes[pos];
};
/** @return {string} JSONified version of BloomSort */
BloomSort.prototype.save = function() {
    // td;
};
exports.BloomSort = BloomSort;
