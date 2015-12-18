var userTraffic = require("../userTraffic");

const testURL = 'http://www.google.com'

exports["test procedure"] = function(assert) {
    userTraffic.start(testURL);
    assert.ok(!userTraffic.isIdle(), 'idle when should not be');
    userTraffic.stop(testURL);
    assert.ok(userTraffic.isIdle(), 'not idle when should be');
}

require("sdk/test").run(exports);
