// gets all links from current page
var links = []

for ( var i = 0; i < document.links.length; i++ ) {
    links.push(document.links[i].href);
}

self.port.emit('links', JSON.stringify(links));
// td: later, links of all frames (via SO answer)

