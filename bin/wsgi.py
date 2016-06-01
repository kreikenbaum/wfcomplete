'''Simple wsgi module which creates a random string of size
'size'. Fails if called incorrectly.'''
import logging
import random
import string

LOGFORMAT='%(levelname)s:%(filename)s:%(lineno)d:%(message)s'
logging.basicConfig(format=LOGFORMAT, level=logging.INFO)


def application(environ, start_response):
    status = '200 OK'
    logging.info('got request: ' + environ['QUERY_STRING'])

    output = _randomString(_reduceSize(_getSize(environ['QUERY_STRING'])))

    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers)

    return [output]

# this depends on if "Proxy-Connection: keep-alive" is included, better safe...
def _reduceSize(size):
    '''reduces size by header length'''
    my_len = size -135 -len(str(size))
    my_len += len(str(size)) - len(str(my_len))
    logging.debug('size: %d', my_len)
    return my_len

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
