"use strict";

const DOC = 'retrieve links from cover traffic';

// gets all links from current page
var links = [];

for ( var i = 0; i < document.links.length; i++ ) {
    if ( !document.links[i].href.contains('file://')
         && !document.links[i].href.contains('127.0.0.1') ) {
	links.push(document.links[i].href);
    }
};

self.port.emit('links', JSON.stringify(links));
// td: later, links of all frames (via SO answer)

