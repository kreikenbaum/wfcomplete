// listens for html calls,
// http://stackoverflow.com/questions/3596583/javascript-detect-an-ajax-event

// to access original values
var s_ajaxListener = new Object();
s_ajaxListener.tempOpen = XMLHttpRequest.prototype.open;
s_ajaxListener.tempSend = XMLHttpRequest.prototype.send;

// called on send()
// td: request type maybe
s_ajaxListener.callback = function () {
    self.port.emit('ajax', getLength());
}

// overrides XMLHttpRequest.prototype.open()
open = function(a,b) {
    if (!a) var a='';
    if (!b) var b='';
    s_ajaxListener.tempOpen.apply(this, arguments);
    delete s_ajaxListener.headers; // new request, clean object
    s_ajaxListener.url = b;
}
exportFunction(open,
	       unsafeWindow.XMLHttpRequest.prototype,
	       {defineAs: "open"});

// overrides XMLHttpRequest.prototype.send()
send = function(a,b) {
    if (!a) var a='';
    if (!b) var b='';
    s_ajaxListener.tempSend.apply(this, arguments);
    s_ajaxListener.callback();
}
exportFunction(send,
	       unsafeWindow.XMLHttpRequest.prototype,
	       {defineAs: "send"});
//td: works without unsafeWindow?


// td: unify access to xmlhttp (either with wrapped or as above)
XMLHttpRequest.prototype.wrappedSetRequestHeader =
  XMLHttpRequest.prototype.setRequestHeader;

// Override the existing setRequestHeader function so that it stores the headers
setRequestHeader = function(header, value) {
    // Call the wrappedSetRequestHeader function first
    // so we get exceptions if we are in an erronous state etc.
    this.wrappedSetRequestHeader(header, value);

    // Create a headers map if it does not exist
    if(!s_ajaxListener.headers) {
        s_ajaxListener.headers = {};
    }

    // Create a list for the header that if it does not exist
    if(!s_ajaxListener.headers[header]) {
        s_ajaxListener.headers[header] = [];
    }

    // Add the value to the header
    s_ajaxListener.headers[header].push(value);
}
exportFunction(setRequestHeader,
	       unsafeWindow.XMLHttpRequest.prototype,
	       {defineAs: "setRequestHeader"});

/** @return the estimated length of this request */
function getLength() {
    var reqLengthCa = s_ajaxListener.url.length;
    if ( s_ajaxListener.headers != undefined ) {
	reqLengthCa += JSON.stringify(s_ajaxListener.headers).length;
    }
    return reqLengthCa;
}
