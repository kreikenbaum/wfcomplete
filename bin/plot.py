'''assorted plotting commands:

- file with org table, 
- list of counters'''
import Gnuplot

COLOR_LIST = ["blue", "green", "yellow", "orange", 'red', "violet"]

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

def _scale(lst, factor=1/1000.0):
    return [x*factor for x in lst]

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

def defenses(places, defs=['disabled/bridge', 'simple1/10'], url='google.com'):
    '''compares defenses at url

    @param defs: defenses to plot
    @param url: url for which to plot defenses

    @return g: otherwise plots are closed ;-)
    '''
    local_list = COLOR_LIST[::len(COLOR_LIST)/len(defs)]
    g = None
    for d in defs:
        g = counters(places[d][url], g, d.replace('/bridge',''),
                     local_list.pop())
    g.title('page: {}'.format(url))
    g.replot()
    return g
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.argv = ['_', 'table']
    g = table(_table_to_data(open(sys.argv[1])))
    raw_input('press any key to exit')
