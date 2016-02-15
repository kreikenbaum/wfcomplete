"""HTTP Server which generates pseudo-random traffic."""

import BaseHTTPServer
import cgi
import random
import string

class TrafficHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    """Server which only generates traffic."""

    def do_POST(self):
        """Does the same thing as GET."""
        try:
            self._gen_traffic(self._find_size_post())
        except (KeyError, ValueError):
            self._fail()

    def do_GET(self):
        """Generate traffic as per size parameter.

        If no size parameter is given, fail.

        """
        try:
            self._gen_traffic(self._find_size_get())
        except (IndexError, ValueError):
            self._fail()

    def _find_size_get(self):
        """Returns the value of the size parameter."""
        paramstring = self.path.split('?')[1]
        for parampair in paramstring.split('&'):
            (var, val) = parampair.split('=')
            if var == 'size':
                return int(val)
        raise IndexError('no size parameter')

    def _find_size_post(self):
        """Returns the value of the size parameter."""
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers.getheader('content-length'))
            postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
        else:
            raise KeyError('wrong input format: ' + ctype)
        return int(postvars['size'])

    def _fail(self):
        """Returns HTTP error message"""
        self.send_error(400, "Bad Request: could not parse the size parameter")

    # td: background thread
    def _gen_traffic(self, size):
        """Generate size bytes of traffic"""
        self.send_response(200)
        self.send_header("Content-Length", size)
        self.end_headers()
        self.wfile.write(''.join(random.choice(string.printable) 
                                 for _ in range(size)))
        
def test(HandlerClass = TrafficHTTPRequestHandler,
         ServerClass = BaseHTTPServer.HTTPServer):
    BaseHTTPServer.test(HandlerClass, ServerClass, protocol="HTTP/1.1")

if __name__ == '__main__':
    test()
