'''assorted plotting commands:

- file with org table, 
- list of counters'''
import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import Gnuplot

def _color_cycle(steps=6):
    '''yields steps of different colors'''
    out = map(_to_color,
              np.linspace(1.0/12, 13.0/12, num=steps, endpoint=False))
    out = map(_to_hex, out)
    return out

def _scale(lst, factor=1/1024.0):
    '''scale list of Bytes to kiloByte'''
    return [x*factor for x in lst]

# old, superseded by _size_table_to_data
def _table_to_data(f):
    '''parse file f with org-mode table export data'''
    Datum = collections.namedtuple('Datum', ['overhead',
                                             'ExtraTrees',
                                             'OneVsRest_SVC'])

    out = {}
    for line in f:
        if "overhead %" in line:
            continue
        (name, et, rf, knn, _, svc, overhead, _) = line.split("\t")
        out[name] = Datum(float(overhead), float(et), float(svc))
    return out

def _gnuplot_ohacc(xrangeleft=0):
    '''@return gnuplot initialized for overhead-accuracy plot'''
    g = Gnuplot.Gnuplot()
    g("set xrange [{}:*]".format(xrangeleft))
    g("set yrange [0:1]")
    g("set xlabel 'overhead [%]'")
    g("set ylabel 'accuracy [%]'")
    return g

def plot_defenses_helper(g, filename='../../data/results/export_30sites.csv'):
    '''plots all defenses in filename'''
    # g = _gnuplot_ohacc()
    with open(filename) as f:
        t = _size_table_to_data(f)
        plot_defenses(g, t)
        for defense in set([x.defense.split('/')[0] for x in t]):
            g.replot(Gnuplot.Data([(x.size, x.cumul) for x in t
                                   if 'plotskip' not in x.notes
                                   and defense in x.defense],
                                  title=defense))

def plot_defenses(g, data):
    '''plots data items, separates by defense'''
    for defense in set([x.defense.split('/')[0] for x in data]):
        g.replot(Gnuplot.Data([(x.size, x.cumul) for x in data
                               if 'plotskip' not in x.notes
                               and defense in x.defense],
                              title=defense))

def plot_flavours(g, data):
    '''plots flavors of main defense'''
    main = [x for x in data if '0.22' in x.defense]
    for flavor in ['aI', 'aII', 'bI', 'bII']:
        its_datas = [x for x in main if '{}-'.format(flavor) in x.defense]
        g.replot(Gnuplot.Data([(x.size, x.cumul) for x in its_datas],
                              title=flavor))

def _replace(stringlist):
    '''removes nonparseable stuff
    >>> replace(['hi', '-', '\open '])
    ['hi', '_', '']
    '''
    out = []
    for s in stringlist:
        out.append(s
                   .replace('-', '_')
                   .replace("\open ", "")
                   .replace(" \close", ""))
    return out

def _size_table_to_data(readable):
    '''read readable data, which contains org-export-table in csv format'''
    read = []
    reader = csv.reader(readable)
    DataRecord = collections.namedtuple('DataRecord', _replace(reader.next()))
    for row in reader:
        read.append(DataRecord._make(row))
    return read

def _to_color(value):
    '''gives color to hls-hue'''
    import colorsys
    return colorsys.hls_to_rgb(value, 0.5, 1)

def _to_hex(a):
    '''color tuple to hex string'''
    return "0x{:02x}{:02x}{:02x}".format(int(255 * a[0]),
                                         int(255 * a[1]),
                                         int(255 * a[2]))

def counters(counter_list, gnuplotter=None, label=None, color="blue"):
    '''counter's cumul data in color. in gnuplotter is not None, do reuse.

    usage: g = plot(COUNTERS['soso.com']);
    g = plot(COUNTERS['msn.com'], color="orange", gnuplotter=g)

    @return gnuplotter for additional plots'''
    if not gnuplotter:
        gnuplotter = Gnuplot.Gnuplot()
        gnuplotter("set xlabel 'Feature Index'")
        gnuplotter("set ylabel 'Feature Value [kByte]'")
    withfmt = 'lines lc rgb "'+color+'"'

    if label: # use any counter separately just to plot the label
        data = [Gnuplot.Data(_scale(counter_list.pop().cumul()[4:]),
                    inline=1, with_=withfmt, title=label)]
    else:
        data = []
    data.extend([Gnuplot.Data(_scale(x.cumul()[4:]), inline=1, with_=withfmt)
                for x in counter_list])
    for d in data:
        gnuplotter.replot(d)
    return gnuplotter

def defenses(places, defs=['disabled/bridge', 'simple1/10'], url='google.com'):
    '''compares defenses at url

    @param defs: defenses to plot
    @param url: url for which to plot defenses

    @return g: otherwise plots are closed ;-)
    '''
    local_list = _color_cycle(len(defs))
    g = None
    for d in defs:
        g = counters(places[d][url], g,
                     d.replace('/bridge',''),
                     local_list.pop())
    g.title('CUMUL-traces on {} after defenses'.format(url))
    g.replot()
    return g

def many_defenses(places, defs=['simple1/10', '22.0/10aI'],
                  urls=['google.com', 'netflix.com', 'tumblr.com']):
    for d in defs:
        for u in urls:
            tmp_defs = ['disabled/bridge']; tmp_defs.append(d)
            g = defenses(places, tmp_defs, u)
            save(g, "{}__{}_vs_disabled".format(u, d.replace('/', '@')))

def roc(fpr, tpr):
    '''@return plot object with roc_curve done, use =.show()= to
display, and =.save(...)= to save'''
    out = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return out

def save(gnuplotter, prefix='plot', type_='eps'):
    '''saves plot to dated file'''
    import time
    gnuplotter.hardcopy("/tmp/mem/{}_{}.{}".format(prefix,
                                                   time.strftime('%F_%T'),
                                                   type_),
                        enhanced=1, color=1, mode="eps")

def table(data, cls="svc"):
    '''plot table data = {defense: {overhead: ..., svc: ...} '''
    g = _gnuplot_ohacc()

    for (defense, datum) in data.items():
        g.replot(Gnuplot.Data([(datum.overhead, datum._asdict()[cls]*100)],
                              title=defense.replace('@', '-'), inline=1),
                 title="method: " + cls.replace('_', ' '))
    return g

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.argv = ['_', 'table']
    data = _table_to_data(open(sys.argv[1]))
    for cls in ['ExtraTrees', 'OneVsRest_SVC']:
        g = table(data, cls)
        save(g, 'defenses with {}'.format(cls))
    raw_input('press the enter key to exit')