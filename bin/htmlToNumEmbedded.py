TAGS = [('audio', 'src'), ('embed', 'src'), ('img', 'src'), ('object', 'data'), ('script', 'src'), ('source', 'src'), ('video', 'src')]
# ('link', 'href')
TESTNAME = '/home/uni/da/git/sw/data/HTMLsize100/alipay.com/index.html'

# td: check urllib for status code
from lxml import html
import urllib
def extractEmbeddedLxml(urlFileOrName):
    done =  False
    while not done:
        try:
            urlFileOrName = urllib.urlopen(urlFileOrName)
            done = True
        except:
            print 'url-opening {} failed, retrying'.format(urlFileOrName)
    xmldoc = html.parse(urlFileOrName)
    root = xmldoc.getroot()
    try:
        root.make_links_absolute()
    except:
        import pdb; pdb.set_trace()
    links = []
    for (tag, attrib) in TAGS:
        for el in root.iter(tag):
            if el.get(attrib) is not None:
                links.append(el.get(attrib))
    for el in root.iter('link'):
        if el.get('rel') in ['stylesheet', 'shortcut icon', 'icon']:
            links.append(el.get('href'))
    return (urlFileOrName.geturl(), links)

# from bs4 import BeautifulSoup
# def extractEmbeddedBeautifulSoup(filename):
#     with open(filename) as f:
#         soup = BeautifulSoup(f)
#     els = []
#     links = []
#     for (tag, attrib) in TAGS:
#         for el in root.iter(tag):
#             els.append(el)
#     return (els, links)

if __name__ == '__main__':
    import sys
    for i in sys.argv[1:]:
        (src, embedded) = extractEmbeddedLxml(i)
        print '{}: {}'.format(src, len(embedded))
