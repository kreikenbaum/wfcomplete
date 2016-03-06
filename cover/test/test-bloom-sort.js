"use strict";

const BloomSort = require("../js/bloom-sort.js");

exports["test add/query"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 3);
    assert.ok(s.query('hi', 5), 'wrong result: ' + s.query('hi', 5));
};

exports["test add/query 2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 8);
    assert.ok(s.query('hi', 15), 'wrong result: ' + s.query('hi', 5));
};

exports["test add/query 2 with different sizes v.2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 18);
    s.add('hi', 3);
    try {
	s.query('hi');
	assert.ok(false, 'did not throw');
    } catch (e) {
	assert.ok(e.message === 'Contains multiple entries');
    }
};
exports["test add/query 2 with different sizes v.1"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 18);
    s.add('hi', 3);
    assert.throws(function() {
	s.query('hi');
    },
		  /multiple entries/,
		  "threw wrong or no exception");
};

exports["test add/query bigger"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15, 25], [10, 20]);
    s.add('hi', 23);
    assert.ok(s.query('hi', 25), 'wrong result: ' + s.query('hi', 5));
};

exports["test creation"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
};

exports["test query empty"] = function(assert) {
    let s = new BloomSort.BloomSort([5,15], [10]);
    assert.throws(function() {
	s.query('not_there');
    },
		  /no entries/,
		  "threw wrong or no exception");
};

require("sdk/test").run(exports);
