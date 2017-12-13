'''assorted plotting commands:

- file with org table,
- list of counters
- roc curve'''
import collections
import csv
import doctest
import numpy as np

import Gnuplot

def _replace(stringlist):
    '''removes nonparseable stuff
    >>> _replace(['hi', '-', r'\open '])
    ['hi', '_', '']
    '''
    out = []
    for s in stringlist:
        out.append(s
                   .replace('-', '_')
                   .replace(r"\open ", "")
                   .replace(r" \close", ""))
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

def _color_cycle(steps=6):
    '''yields steps of different colors
    >>> _color_cycle(3)
    ['0xff7f00', '0x00ff7f', '0x7f00ff']
    '''
    return [_to_hex(_to_color(x)) for x in
            np.linspace(1.0/12, 13.0/12, num=steps, endpoint=False)]

def _scale(lst, factor=1/1024.0):
    '''scale list of Bytes to kiloByte'''
    return [x*factor for x in lst]

# # old, superseded by _size_table_to_data
# def _table_to_data(f):
#     '''parse file f with org-mode table export data'''
#     Datum = collections.namedtuple('Datum', ['overhead',
#                                              'ExtraTrees',
#                                              'OneVsRest_SVC'])

#     out = {}
#     for line in f:
#         if "overhead %" in line:
#             continue
#         (name, et, rf, knn, _, svc, overhead, _) = line.split("\t")
#         out[name] = Datum(float(overhead), float(et), float(svc))
#     return out

def _gnuplot_ohacc(xrangeleft=0):
    '''@return gnuplot initialized for overhead-accuracy plot'''
    plot = Gnuplot.Gnuplot()
    plot("set xrange [{}:*]".format(xrangeleft))
    plot("set yrange [0:1]")
    plot("set xlabel 'overhead [%]'")
    plot("set ylabel 'accuracy [%]'")
    return plot

def plot_defenses_helper(plot,
                         filename='../../data/results/export_30sites.csv'):
    '''plots all defenses in filename'''
    # g = _gnuplot_ohacc()
    with open(filename) as f:
        t = _size_table_to_data(f)
        plot_defenses(plot, t)
        for defense in set([x.defense.split('/')[0] for x in t]):
            plot.replot(Gnuplot.Data([(x.size, x.cumul) for x in t
                                      if 'plotskip' not in x.notes
                                      and defense in x.defense],
                                     title=defense))

def plot_defenses(plot, data):
    '''plots data items, separates by defense'''
    for defense in set([x.defense.split('/')[0] for x in data]):
        plot.replot(Gnuplot.Data([(x.size, x.cumul) for x in data
                                  if 'plotskip' not in x.notes
                                  and defense in x.defense],
                                 title=defense))

def plot_flavours(plot, data):
    '''plots flavors of main defense'''
    main = [x for x in data if '0.22' in x.defense]
    for flavor in ['aI', 'aII', 'bI', 'bII']:
        its_datas = [x for x in main if '{}-'.format(flavor) in x.defense]
        plot.replot(Gnuplot.Data([(x.size, x.cumul) for x in its_datas],
                                 title=flavor))

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
                     d.replace('/bridge', ''),
                     local_list.pop())
    g.title('CUMUL-traces on {} after defenses'.format(url))
    g.replot()
    return g

def many_defenses(places, defs=['simple1/10', '22.0/10aI'],
                  urls=['google.com', 'netflix.com', 'tumblr.com']):
    '''? prints many defenses to separate plot files'''
    for d in defs:
        for u in urls:
            tmp_defs = ['disabled/bridge']; tmp_defs.append(d)
            plot = defenses(places, tmp_defs, u)
            save(plot, "{}__{}_vs_disabled".format(u, d.replace('/', '@')))

def save(gnuplotter, prefix='plot', type_='eps'):
    '''saves plot to dated file'''
    import time
    gnuplotter.hardcopy("/tmp/mem/{}_{}.{}".format(prefix,
                                                   time.strftime('%F_%T'),
                                                   type_),
                        enhanced=1, color=1, mode="eps")

def table(data, cls="svc"):
    '''plot table data = {defense: {overhead: ..., svc: ...} '''
    plot = _gnuplot_ohacc()

    for (defense, datum) in data.items():
        plot.replot(Gnuplot.Data([(datum.overhead, datum._asdict()[cls]*100)],
                                 title=defense.replace('@', '-'), inline=1),
                    title="method: " + cls.replace('_', ' '))
    return plot

doctest.testmod()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         sys.argv = ['_', 'table']
#     data = _table_to_data(open(sys.argv[1]))
#     for cls in ['ExtraTrees', 'OneVsRest_SVC']:
#         g = table(data, cls)
#         save(g, 'defenses with {}'.format(cls))
#     raw_input('press the enter key to exit')
