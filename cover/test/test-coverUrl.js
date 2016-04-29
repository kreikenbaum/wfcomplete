"use strict";

const Simple = require('sdk/simple-prefs');

const coverUrl = require("../js/coverUrl.js");

var HOST = Simple.prefs['Traffic-HOST'];
var PORT = Simple.prefs['Traffic-PORT'];

exports["test contains"] = function(assert) {
    assert.ok(coverUrl.contains(HOST), 'no contain');
};
exports["test contains param"] = function(assert) {
    assert.ok(coverUrl.contains(HOST + ":" + PORT + '/?size=1234'),
	      'no contain param');
};
exports["test contains uncontained"] = function(assert) {
    assert.ok(! coverUrl.contains('http://somewhere.com'), 'contains');
};


exports["test sized"] = function(assert) {
    var result = coverUrl.sized(10);
    assert.equal(result, 'http://' + HOST + ":" + PORT + '/?size=10',
		 'failed: ' + result);
};

require("sdk/test").run(exports);
