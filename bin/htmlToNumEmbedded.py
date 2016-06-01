#! /usr/bin/env python
'''retrieves url, looks for for number of embedded elements'''
import logging
import urllib2

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

TAGS = [('audio', 'src'), ('embed', 'src'), ('img', 'src'), ('object', 'data'), ('script', 'src'), ('source', 'src'), ('video', 'src')]
# ('link', 'href')
TESTNAME = '/home/uni/da/git/sw/data/HTMLsize100/alipay.com/index.html'

interrupted = False

# td: check urllib for status code
from lxml import html
def extract_embedded(url_file_or_name):
    done =  False
    while not done and not interrupted:
        req = urllib2.Request(url_file_or_name,
                              headers={'User-Agent': 'Mozilla/5.0 '
    + '(X11; Ubuntu; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0' })
        try:
            url_file_or_name = urllib2.urlopen(req)
            done = True
        except:
            logging.warn('url-opening {} failed'.format(url_file_or_name))
    xmldoc = html.parse(url_file_or_name)
    root = xmldoc.getroot()
    root.make_links_absolute()
    links = []
    for (tag, attrib) in TAGS:
        for el in root.iter(tag):
            if el.get(attrib) is not None:
                links.append(el.get(attrib))
    for el in root.iter('link'):
        if el.get('rel') in ['stylesheet', 'shortcut icon', 'icon']:
            links.append(el.get('href'))
    return (url_file_or_name.geturl(), links)

if __name__ == '__main__':
    import sys
    for i in sys.argv[1:]:
        logging.debug('trying {}'.format(i))
        try:
            (src, embedded) = extract_embedded(i)
        except KeyboardInterrupt:
            interrupted = True
        logging.info('embedded: {}'.format(embedded))
        print '{}: {}'.format(src, len(set(embedded)))
