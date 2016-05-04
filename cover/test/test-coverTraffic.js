"use strict";
const coverTraffic = require("../js/coverTraffic.js");
const mockLoad = require("./mock-load.js");

const HOSTNAME = 'http://unknown.host';

exports["test object"] = function(assert) {
    mockLoad.reset();
    assert.equal(mockLoad.getCount(), 0, 'initialization error');
    var ct = new coverTraffic.CoverTraffic(HOSTNAME, mockLoad);
    assert.equal(mockLoad.getCount(), 1, 'no calls to mock');
    for ( var i = 20; i > 0; i -= 1 ) {
	ct.loadNext();
    }
    assert.ok(mockLoad.getCount() >= 2, 
	      'no calls to mock on ct.loadNext(): ' + mockLoad.getCount());
};

exports["test two objects"] = function(assert) {
    mockLoad.reset();
    assert.equal(mockLoad.getCount(), 0, 'initialization error');
    var ct = new coverTraffic.CoverTraffic(HOSTNAME, mockLoad);
    assert.equal(mockLoad.getCount(), 1, 'no call to mock of ct');
    var c2 = new coverTraffic.CoverTraffic(HOSTNAME, mockLoad);
    assert.equal(mockLoad.getCount(), 2, 'no call to mock of ct2');
    for ( let i = 10; i > 0; i -= 1 ) {
	ct.loadNext();
    }
    for ( let i = 10; i > 0; i -= 1 ) {
	c2.loadNext();
    }
    assert.ok(mockLoad.getCount() >= 4, 
	      'no calls to mock on ct.loadNext() or c2.loadNext(): ' +
	      mockLoad.getCount());
};

exports["test object only in htmlCache (strategy 1=IA)"] = function(assert) {
    const GURL = "http://google.com/";
    mockLoad.reset();
    assert.equal(mockLoad.getCount(), 0, 'initialization error');
    var ct = new coverTraffic.CoverTraffic(GURL, mockLoad);
    assert.equal(mockLoad.getSum(), 266);
};

// exports["test failure without loader"] = function(assert) {
//     coverTraffic.setLoader(null);
//     try {
// 	coverTraffic.start();
// 	assert.ok(false, 'loader = null raised no error');
//     } catch(e) {
// 	// all's well
//     }
// };

require("sdk/test").run(exports);
