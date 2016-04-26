"use strict";

//const {Cc, Ci} = require("chrome");

const { URL } = require("sdk/url");

//const ioService = Cc["@mozilla.org/network/io-service;1"]
//		    .getService(Ci.nsIIOService);

const user = require("../js/user.js");

const TEST = URL('http://www.google.com/search');

// exports["test single"] = function(assert) {
//     user.loads(ioService.newURI('http://www.test.de', null, null));
//     assert.ok(! user.isIdle(), 'idle after start');
//     user.endsLoading(ioService.newURI('http://www.test.de', null, null));
//     assert.ok(user.isIdle(), 'not idle after stop');
// }

// exports["test double"] = function(assert) {
//     user.loads(ioService.newURI('http://www.test.de', null, null));
//     assert.ok(! user.isIdle(), 'idle after start');
//     user.loads(ioService.newURI('http://www.test.de/pic.png', null, null));
//     assert.ok(! user.isIdle(), 'idle after picture');
//     user.endsLoading(ioService.newURI('http://www.test.de', null, null));
//     assert.ok(user.isIdle(), 'not idle after stop');
// }

exports["test stripParam"] = function(assert) {
    console.log("url: " + TEST.href);
    assert.equal(user.stripParam(URL(TEST.href + '?asdf')).href, TEST.href);
    assert.equal(user.stripParam(TEST).href, TEST.href);
    assert.equal(user.stripParam(URL(TEST.href + '#test')).href, TEST.href);
}

//     user.loads(ioService.newURI('http://www.test.de', null, null));
//     assert.ok(! user.isIdle(), 'idle after start');


require("sdk/test").run(exports);
