"use strict";

const BloomSort = require("../js/bloom-sort.js");

exports["test add/query"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 3);
    assert.equal(s.query('hi'), 5, 'wrong result: ' + s.query('hi'));
};

exports["test add/query 2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 8);
    assert.equal(s.query('hi'), 5, 'wrong result: ' + s.query('hi'));
};

exports["test add/queryMax"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 3);
    assert.equal(s.queryMax('hi'), 10, 'wrong result: ' + s.queryMax('hi'));
};

exports["test add/queryMax 2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 12);
    let res = s.queryMax('hi');
    assert.equal(res, 15);
};

exports["test add/query 2 with different sizes v.2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 18);
    s.add('hi', 3);
    try {
	s.query('hi');
	assert.ok(false, 'did not throw');
    } catch (e) {
	assert.equal(e.message, 'Contains multiple entries');
    }
};
exports["test add/query 2 with different sizes v.1"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 18);
    s.add('hi', 3);
    assert.throws(function() { s.query('hi'); },
		  /multiple entries/,
		  "threw wrong or no exception");
};

exports["test add/query bigger"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15, 25], [10, 20]);
    s.add('hi', 23);
    assert.equal(s.query('hi'), 25, 'wrong result: ' + s.query('hi'));
};

exports["test creation"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
};

exports["test add/query 2"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 8);
    assert.equal(s.query('hi'), 5, 'wrong result: ' + s.query('hi'));
};

exports["test save/restore"] = function(assert) {
    let s = new BloomSort.BloomSort([5, 15], [10]);
    s.add('hi', 8);
    s.save();
    let t = BloomSort.restore();
    try {
        assert.equal(t.query('hi'), 5, 'wrong result: ' + t.query('hi'));
    } catch (e) {
        assert.ok(false, 'failed with exception (' + e + '): ' + t);
    }
};


exports["test query empty"] = function(assert) {
    let s = new BloomSort.BloomSort([5,15], [10]);
    assert.throws(function() { s.query('not_there'); },
		  /no entries/,
		  "threw wrong or no exception");
};

require("sdk/test").run(exports);
