/*
const { NetUtil } = require("resource://gre/modules/NetUtil.jsm"); 
const { Services: { obs } } = require("resource://gre/modules/Services.jsm");

const events = require("sdk/system/events");

const subject = require("./mock-subject.js");

// test that sending a http-on-modify-request triggers watcher
exports["test event reaction"] = function(assert) {
    let watcher = require("../js/watcher.js");
    let mockUser = require("./mock-user.js"); 
    assert.equal(mockUser.loaded.length, 0, 'initialization error');
    watcher.register(mockUser);
    // td here: send event via obs
    // obs.notifyObservers(subject.subject,
    // 	// { QueryInterface : function() {
    // 	//     return {"URI": NetUtil.newURI("mock://URI.spec") };
    // 	// }},
    // 	"http-on-modify-request",
    // 	null);
    // or via events
    events.emit("http-on-modify-request", {"subject": subject.subject,
					   "data": "test data"});
    assert.equal(mockUser.loaded.length, 1, 'error loading one');
};
*/

require("sdk/test").run(exports);
