"use strict";

var userTraffic = require("../userTraffic");

exports["test procedure"] = function(assert) {
    const url = 'http://www.test.de';
    userTraffic.start(url);
    assert.ok(! userTraffic.isIdle(), 'idle after start');
    userTraffic.stop(url);
    assert.ok(userTraffic.isIdle(), 'not idle after stop');
}

require("sdk/test").run(exports);
