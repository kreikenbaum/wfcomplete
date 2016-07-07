"use strict";

const User = require("../js/user.js");

let user = new User.User();

const mock = require("./mock.js");

const TESTURL = new mock.URL('http://testurl.com');
const TESTURL2 = new mock.URL('http://testurl2.com');
const TESTURL3 = new mock.URL('http://testurl3.com');

exports["test load+finish once"] = function(assert) {
    assert.ok(! user.loads(TESTURL));
    assert.ok(user.endsLoading(TESTURL));
};

exports["test finish twice"] = function(assert) {
    user.loads(TESTURL2);
    assert.ok(user.endsLoading(TESTURL2));
    assert.ok(! user.endsLoading(TESTURL2));
};

exports["test load twice"] = function(assert) {
    assert.ok(! user.loads(TESTURL3));
    assert.ok(user.loads(TESTURL3));
    assert.ok(user.endsLoading(TESTURL3));
};

require("sdk/test").run(exports);
