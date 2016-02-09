"use strict";

const {Cc, Ci} = require("chrome");

const user = require("../js/user.js");
const ioService = Cc["@mozilla.org/network/io-service;1"]
      .getService(Ci.nsIIOService);

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

require("sdk/test").run(exports);
