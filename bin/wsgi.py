'''Simple wsgi module which creates a random string of size
'size'. Fails if called incorrectly.'''
import logging
import random
import string

def application(environ, start_response):
    status = '200 OK'
    logging.warning('got request with ' + environ['QUERY_STRING'])

    output = _randomString(_getSize(environ['QUERY_STRING']))

    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)

    return [output]


def _getSize(query):
    '''extracts value of size parameter'''
    for parampair in query.split('&'):
        (var, val) = parampair.split('=')
        if var == 'size':
            return max(1, int(val))
    raise IndexError('no size parameter')

def _randomString(size):
    '''generate random string of size "size"'''
    return ''.join(random.choice(string.printable) for _ in range(size))
