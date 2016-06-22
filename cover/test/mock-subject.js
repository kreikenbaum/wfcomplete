"use strict";
/**
* @fileoverview provides mock for event.subject
*/
const { Class } = require('sdk/core/heritage');
const { Unknown } = require('sdk/platform/xpcom');

let subject = Class({
    extends: Unknown,
    interfaces: [ 'nsIHttpChannel' ],
    URI: {"spec": 'mock://URI.spec'}
});
		    
exports.subject = subject; // {
//     QueryInterface : function() {
// 	return {"URI": {"spec": 'mock://URI.spec'}};
//     }
// };
