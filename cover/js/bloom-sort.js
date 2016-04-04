"use strict";
/**
* @fileoverview uses bloomfilters to sort elements
*/
const _ = require('../lib/underscore-min.js');
const Bloom = require("../lib/bloomfilter.js");
const Storage = require("sdk/simple-storage").storage;

const debug = require("./debug.js");

// td later (in thesis):choose parameters 
const NUM_BITS = 32 * 256; // number of bits to allocate.
const NUM_HASH = 3; 

// td: typeerror if sizes.length !== splits.length + 1
// td: maybe test if sizes and splits are sorted and arrayed correctly
// td: generate splits if empty

/** 
 * @param {sizes Array} array of values for each bucket, must be sorted
 * @param {splits Array} array of bucket borders, must be sorted
*/
function BloomSort(sizes, splits) {
    this.sizes = sizes;
    this.splits = splits; // || determine_splits(sizes);
    this.filters = [];
    for ( let i = 0; i < sizes.length; i++ ) {
	this.filters[i] = new Bloom.BloomFilter(NUM_BITS, NUM_HASH); 
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
/** saves BloomSort-array to local storage */
BloomSort.prototype.save = function() {
    Storage.filters = [];
    for ( let i = 0; i < this.filters.length; i++ ) {
        Storage.filters[i] = [].slice.call(this.filters[i].buckets);
    }
    Storage.splits = this.splits;
    Storage.sizes = this.sizes;
};
BloomSort.prototype.toString = function() {
    let out = 'BloomSort = {\n';
    out += 'sizes: ' + this.sizes + '\n';
    out += 'splits: ' + this.splits + '\n';
    out += 'filters: ' + this.filters + '\n';
    out += '}';
    return out;
};
exports.BloomSort = BloomSort;

/** @return bloomSort restored from local storage */
function restore() {
    let s = new BloomSort(Storage.sizes, Storage.splits);
    for ( let i = 0; i < s.filters.length; i++ ) {
	s.filters[i] = new Bloom.BloomFilter(Storage.filters[i], NUM_HASH);
    }
    return s;
}
exports.restore = () => restore();
