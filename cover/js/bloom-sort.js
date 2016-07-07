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
 * sorts elements into bins limited by splits, assigns values in sizes
 * @constructor
 * @param {sizes Array} array of values for each bin, must be sorted
 * @param {splits Array} array of bin borders, must be sorted
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
    //console.log('add(' + id + ', ' + size + ')');
    this.filters[_.sortedIndex(this.splits, size)].add(id);
};
/** @return size of element, raises {@code BloomError} if unclear */
BloomSort.prototype.query = function(id) {
    return this.sizes[this.getPosition(id)];
};
/** @return {Number} the upper border of the bin in which id is
 * found. If biggest bin, return its {@code size}. */
BloomSort.prototype.queryMax = function(id) {
    let pos = this.getPosition(id);
    if ( pos < this.splits.length ) {
	return this.splits[pos];
    } else { // pos == this.splits.length
	return this.sizes[pos];
    }
};
BloomSort.prototype.getPosition = function(id) {
    //    console.log('getPosition(' + id +")");
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
	    message: 'Contains no entries matching: ' + id
	};
    }
    return pos;
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
