'''assorted plotting commands:

- file with org table, 
- list of counters'''
import Gnuplot

import numpy as np

def _color_cycle(steps=6):
    '''yields steps of different colors'''
    out = map(_to_color,
              np.linspace(1.0/12, 13.0/12, num=steps, endpoint=False))
    out = map(_to_hex, out)
    return out

def _scale(lst, factor=1/1024.0):
    '''scale byte to kByte'''
    return [x*factor for x in lst]

def _table_to_data(f):
    '''parse file f with org-mode table export data'''
    import collections

    Datum = collections.namedtuple('Datum', ['overhead', 'svc', 'extratrees'])

    out = {}
    for line in f:
        if "overhead (in %)" in line:
            continue
        (name, et, _, _, _, svc, overhead, _) = line.split("\t")
        out[name] = Datum(float(overhead), float(svc), float(et))
    return out

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
        g = counters(places[d][url], g, d.replace('/bridge',''),
                     local_list.pop())
    g.title('CUMUL-traces on {} after defenses'.format(url))
    g.replot()
    return g

def save(gnuplotter, prefix='plot'):
    '''saves plot to dated file'''
    import time
    gnuplotter.hardcopy("/tmp/mem/{}_{}.eps".format(prefix,
                                                    time.strftime('%F_%T')),
                        enhanced=1, color=1)

def table(data, cls="svc"):
    '''table data'''
    g = Gnuplot.Gnuplot()
    g("set xrange [0:*]")
    g("set yrange [0:100]")
    g("set xlabel 'overhead [\%]'")
    g("set ylabel 'accuracy [\%]'")

    for (defense, datum) in data.items():
        g.replot(Gnuplot.Data([(datum.overhead, datum._asdict()[cls]*100)],
                              title=defense, inline=1), title="method: " + cls)
    return g

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.argv = ['_', 'table']
    g = table(_table_to_data(open(sys.argv[1])))
    raw_input('press any key to exit')
