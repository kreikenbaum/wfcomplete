"use strict";

const NAME = 'coverUrl';
exports.DOC = 'database of url and sizes';

const _ = require("lib/underscore-min.js");
const debug = require("./debug.js");
const random = require("js/random.js");
const stats = require("./stats.js");
// for debugging, use //var _ = require("../underscore.js");

// TODO: (after it works) implement real URLs, not mlsec placeholders
// TODO: (after it works) add urls and sizes from *cover* traffic
// TD: include bloom filter for real traffic urls (do not store, but bloom)
// (only cover, not from user traffic), save at end, restore

var contains = function(url) {
    var index = URLS.indexOf(url);
    debug.traceLog(NAME + ': url: ' + url
		   + ' indexof: ' + index  + ' return: ' + (index !== -1), this);
    return index !== -1;
};
exports.contains = url => contains(url); // td: obsolete this, then rename
exports.includes = url => contains(url);

/** provides a url for an object with approximately the given size
 * @return {string} a URL pointing to object of more than size */
var sized = function(size) {
    if ( size > _.last(SIZES) ) {
	debug.log("size " + size + " bigger than available urls");
	return pad(_.last(URLS));
    }
    if ( size < SIZES[0] ) {
	debug.log("size " + size + " smaller than available urls");
	return pad(URLS[0]);
    }

    // pad with random string
    return pad(URLS[_.sortedIndex(SIZES, size)])
};
exports.sized = size => sized(size);


// PRIVATE
/** @return {string} GET request url padded with random characters */
var pad = function(url) {
    var separator = ( _.contains(url, '?') ) ? '&' : '?';
    return url + separator + random.string(stats.requestLength() - url.length);
}

const SIZES = [27, 86, 86, 95, 95, 95, 98, 98, 98, 142, 142, 142, 147, 147, 147, 148, 151, 153, 153, 153, 169, 169, 169, 177, 177, 177, 184, 184, 184, 216, 229, 230, 237, 242, 245, 246, 256, 307, 309, 329, 348, 356, 453, 458, 468, 493, 585, 597, 597, 600, 616, 616, 625, 676, 676, 676, 746, 746, 825, 856, 937, 1038, 1054, 1082, 1110, 1110, 1110, 1163, 1163, 1163, 1301, 1327, 1353, 1353, 1353, 1423, 1423, 1423, 1423, 1423, 1423, 1423, 1423, 1637, 1673, 1754, 1754, 1756, 1910, 1961, 1987, 1987, 2118, 2118, 2118, 2118, 2118, 2118, 2118, 2118, 2344, 2441, 2445, 2449, 2484, 2565, 2567, 2567, 2567, 2567, 2567, 2581, 2983, 3140, 3140, 3154, 3154, 3154, 3154, 3154, 3154, 3154, 3154, 3182, 3229, 3242, 3242, 3284, 3284, 3289, 3310, 3372, 3393, 3443, 3466, 3466, 3485, 3532, 3532, 3634, 3634, 3634, 3730, 3764, 3765, 3779, 3779, 3779, 3812, 3979, 3979, 3979, 3979, 3979, 3979, 3979, 3979, 4175, 4247, 4342, 4369, 4369, 4393, 4393, 4410, 4459, 4482, 4499, 4774, 4775, 4777, 4841, 4871, 4895, 4919, 4955, 4955, 4963, 5025, 5052, 5054, 5069, 5207, 5207, 5270, 5293, 5328, 5344, 5369, 5444, 5504, 5504, 5516, 5516, 5629, 5702, 5827, 5889, 5891, 5936, 5959, 6068, 6072, 6121, 6185, 6192, 6201, 6217, 6222, 6269, 6433, 6452, 6454, 6482, 6537, 6774, 6946, 6950, 6961, 7001, 7018, 7021, 7027, 7031, 7045, 7460, 7767, 7853, 7940, 8147, 8266, 8731, 8769, 8775, 8780, 8790, 8891, 8921, 8979, 9374, 9578, 9861, 10141, 10240, 10278, 10910, 10928, 10981, 11108, 11222, 11576, 11988, 12208, 12315, 15049, 15721, 16115, 16535, 17824, 20929, 21330, 21877, 23093, 23854, 25495, 25495, 25566, 25901, 26141, 27325, 27738, 27971, 28141, 28497, 28997, 30706, 33312, 33673, 33789, 36300, 37296, 37883, 38644, 38884, 39679, 39745, 39790, 40611, 40884, 41286, 41428, 42448, 43634, 45400, 45655, 45681, 47433, 49145, 50000, 55122, 56385, 57728, 64527, 65633, 68381, 72358, 72513, 74387, 75207, 78484, 79421, 84807, 85055, 87005, 88834, 98989, 102315, 102421, 104693, 108505, 125802, 126352, 128304, 135962, 138051, 146073, 146338, 146338, 208549, 216262, 224864, 230677, 249130, 272214, 272254, 282961, 290576, 292379, 343990, 364042, 370756, 372130, 376092, 407348, 413354, 496288, 497825, 501515, 501534, 510092, 510153, 511024, 511191, 511206, 512903, 512916, 521654, 524435, 534262, 536386, 544824, 547312, 551306, 551697, 552164, 552177, 554416, 559399, 570946, 574476, 582201, 591824, 592112, 594805, 618280, 619657, 773133, 800307, 1113652, 1129057, 1139814, 1188090, 1459948, 2587080];

const URLS = ['http://mlsec.org/robots.txt', 'http://mlsec.org/malheur/api/ftv2lastnode.png', 'http://mlsec.org/malheur/api/ftv2node.png', 'http://mlsec.org/harry/api/nav_g.png', 'http://mlsec.org/malheur/api/nav_g.png', 'http://mlsec.org/sally/api/nav_g.png', 'http://mlsec.org/harry/api/nav_h.png', 'http://mlsec.org/malheur/api/nav_h.png', 'http://mlsec.org/sally/api/nav_h.png', 'http://mlsec.org/harry/api/tab_a.png', 'http://mlsec.org/malheur/api/tab_a.png', 'http://mlsec.org/sally/api/tab_a.png', 'http://mlsec.org/harry/api/bdwn.png', 'http://mlsec.org/malheur/api/bdwn.png', 'http://mlsec.org/sally/api/bdwn.png', 'http://mlsec.org/icons/blank.gif', 'http://mlsec.org/harry/examples/data.txt', 'http://mlsec.org/harry/api/nav_f.png', 'http://mlsec.org/malheur/api/nav_f.png', 'http://mlsec.org/sally/api/nav_f.png', 'http://mlsec.org/harry/api/tab_b.png', 'http://mlsec.org/malheur/api/tab_b.png', 'http://mlsec.org/sally/api/tab_b.png', 'http://mlsec.org/harry/api/tab_h.png', 'http://mlsec.org/malheur/api/tab_h.png', 'http://mlsec.org/sally/api/tab_h.png', 'http://mlsec.org/harry/api/tab_s.png', 'http://mlsec.org/malheur/api/tab_s.png', 'http://mlsec.org/sally/api/tab_s.png', 'http://mlsec.org/icons/back.gif', 'http://mlsec.org/icons/text.gif', 'http://mlsec.org/salad/salad.css', 'http://mlsec.org/icons/p.gif', 'http://mlsec.org/icons/script.gif', 'http://mlsec.org/icons/unknown.gif', 'http://mlsec.org/icons/binary.gif', 'http://mlsec.org/harry/examples/alexa/run_example.sh', 'http://mlsec.org/joern/joern.bib', 'http://mlsec.org/icons/image2.gif', 'http://mlsec.org/malheur/docs/malheur.bib', 'http://mlsec.org/harry/examples/alexa/most_similar.py', 'http://mlsec.org/sally/docs/sally.bib', 'http://mlsec.org/malheur/api/ftv2cl.png', 'http://mlsec.org/salad/docs/salad.bib', 'http://mlsec.org/salad/util.js', 'http://mlsec.org/harry/examples/reuters/run_example.sh', 'http://mlsec.org/sally/benchmark/results/sally.time', 'http://mlsec.org/harry/api/folderopen.png', 'http://mlsec.org/sally/api/folderopen.png', 'http://mlsec.org/sally/benchmark/results/python.time', 'http://mlsec.org/harry/api/folderclosed.png', 'http://mlsec.org/sally/api/folderclosed.png', 'http://mlsec.org/sally/benchmark/results/matlab.time', 'http://mlsec.org/harry/api/bc_s.png', 'http://mlsec.org/malheur/api/bc_s.png', 'http://mlsec.org/sally/api/bc_s.png', 'http://mlsec.org/harry/api/doc.png', 'http://mlsec.org/sally/api/doc.png', 'http://mlsec.org/sally/examples/example3.cfg', 'http://mlsec.org/sally/examples/example1.cfg', 'http://mlsec.org/sally/examples/example2.cfg', 'http://mlsec.org/icons/compressed.gif', 'http://mlsec.org/harry/examples/reuters/index.html', 'http://mlsec.org/harry/examples/reuters/index.html?C=N;O=D', 'http://mlsec.org/harry/examples/reuters/index.html?C=D;O=A', 'http://mlsec.org/harry/examples/reuters/index.html?C=M;O=A', 'http://mlsec.org/harry/examples/reuters/index.html?C=S;O=A', 'http://mlsec.org/harry/api/tabs.css', 'http://mlsec.org/malheur/api/tabs.css', 'http://mlsec.org/sally/api/tabs.css', 'http://mlsec.org/harry/examples/alexa/index.html', 'http://mlsec.org/harry/examples/alexa/index.html?C=N;O=D', 'http://mlsec.org/harry/examples/alexa/index.html?C=D;O=A', 'http://mlsec.org/harry/examples/alexa/index.html?C=M;O=A', 'http://mlsec.org/harry/examples/alexa/index.html?C=S;O=A', 'http://mlsec.org/harry/files/index.html?C=D;O=A', 'http://mlsec.org/harry/files/index.html?C=D;O=D', 'http://mlsec.org/harry/files/index.html?C=M;O=A', 'http://mlsec.org/harry/files/index.html?C=M;O=D', 'http://mlsec.org/harry/files/index.html?C=N;O=A', 'http://mlsec.org/harry/files/index.html?C=N;O=D', 'http://mlsec.org/harry/files/index.html?C=S;O=A', 'http://mlsec.org/harry/files/index.html?C=S;O=D', 'http://mlsec.org/sally/benchmark/matlab.m', 'http://mlsec.org/sally/benchmark/python.p', 'http://mlsec.org/harry/api/index.html', 'http://mlsec.org/sally/api/index.html', 'http://mlsec.org/malheur/api/index.html', 'http://mlsec.org/harry/api/group__normalization.html', 'http://mlsec.org/salad/images/package_48.png', 'http://mlsec.org/malheur/images/tub-logo.gif', 'http://mlsec.org/sally/images/tub-logo.gif', 'http://mlsec.org/malheur/files/index.html?C=D;O=A', 'http://mlsec.org/malheur/files/index.html?C=D;O=D', 'http://mlsec.org/malheur/files/index.html?C=M;O=A', 'http://mlsec.org/malheur/files/index.html?C=M;O=D', 'http://mlsec.org/malheur/files/index.html?C=N;O=A', 'http://mlsec.org/malheur/files/index.html?C=N;O=D', 'http://mlsec.org/malheur/files/index.html?C=S;O=A', 'http://mlsec.org/malheur/files/index.html?C=S;O=D', 'http://mlsec.org/sally/api/structstring__t.html', 'http://mlsec.org/harry/api/structinput__t.html', 'http://mlsec.org/sally/api/structstoptoken__t.html', 'http://mlsec.org/harry/api/structoutput__t.html', 'http://mlsec.org/harry/api/structstoptoken__t.html', 'http://mlsec.org/sally/api/structfunc__t.html', 'http://mlsec.org/harry/images/ugo-logo.gif', 'http://mlsec.org/joern/images/ugo-logo.gif', 'http://mlsec.org/malheur/images/ugo-logo.gif', 'http://mlsec.org/salad/images/ugo-logo.gif', 'http://mlsec.org/sally/images/ugo-logo.gif', 'http://mlsec.org/salad/files/salad-0.6.0-linux-x86_64.rpm', 'http://mlsec.org/malheur/api/dynsections.js', 'http://mlsec.org/harry/api/dynsections.js', 'http://mlsec.org/sally/api/dynsections.js', 'http://mlsec.org/sally/files/index.html?C=D;O=A', 'http://mlsec.org/sally/files/index.html?C=D;O=D', 'http://mlsec.org/sally/files/index.html?C=M;O=A', 'http://mlsec.org/sally/files/index.html?C=M;O=D', 'http://mlsec.org/sally/files/index.html?C=N;O=A', 'http://mlsec.org/sally/files/index.html?C=N;O=D', 'http://mlsec.org/sally/files/index.html?C=S;O=A', 'http://mlsec.org/sally/files/index.html?C=S;O=D', 'http://mlsec.org/impressum.html', 'http://mlsec.org/images/bag2_16.png', 'http://mlsec.org/images/home_16.png', 'http://mlsec.org/joern/images/home_16.png', 'http://mlsec.org/images/down_16.png', 'http://mlsec.org/joern/images/down_16.png', 'http://mlsec.org/malheur/images/uma-logo.gif', 'http://mlsec.org/images/right_16.png', 'http://mlsec.org/images/bubble_16.png', 'http://mlsec.org/images/letter_16.png', 'http://mlsec.org/images/settings_16.png', 'http://mlsec.org/images/info_16.png', 'http://mlsec.org/joern/images/info_16.png', 'http://mlsec.org/images/clock_16.png', 'http://mlsec.org/images/bug_16.png', 'http://mlsec.org/joern/images/bug_16.png', 'http://mlsec.org/harry/images/idalab-logo.gif', 'http://mlsec.org/salad/images/idalab-logo.gif', 'http://mlsec.org/sally/images/idalab-logo.gif', 'http://mlsec.org/harry/api/structrange__t.html', 'http://mlsec.org/harry/examples/reuters/harry.cfg', 'http://mlsec.org/sally/api/structtoken__t.html', 'http://mlsec.org/harry/api/doxygen.png', 'http://mlsec.org/malheur/api/doxygen.png', 'http://mlsec.org/sally/api/doxygen.png', 'http://mlsec.org/sally/api/modules.html', 'http://mlsec.org/salad/files/index.html?C=D;O=A', 'http://mlsec.org/salad/files/index.html?C=D;O=D', 'http://mlsec.org/salad/files/index.html?C=M;O=A', 'http://mlsec.org/salad/files/index.html?C=M;O=D', 'http://mlsec.org/salad/files/index.html?C=N;O=A', 'http://mlsec.org/salad/files/index.html?C=N;O=D', 'http://mlsec.org/salad/files/index.html?C=S;O=A', 'http://mlsec.org/salad/files/index.html?C=S;O=D', 'http://mlsec.org/sally/api/annotated.html', 'http://mlsec.org/sally/api/functions.html', 'http://mlsec.org/harry/api/structhmatrixspec__t.html', 'http://mlsec.org/images/user_info_48.png', 'http://mlsec.org/joern/images/user_info_48.png', 'http://mlsec.org/images/label2_32.png', 'http://mlsec.org/joern/images/label2_32.png', 'http://mlsec.org/harry/depend.html', 'http://mlsec.org/harry/api/modules.html', 'http://mlsec.org/harry/api/structmeasure__t.html', 'http://mlsec.org/salad/depend.html', 'http://mlsec.org/harry/api/annotated.html', 'http://mlsec.org/salad/examples.html', 'http://mlsec.org/harry/examples.html', 'http://mlsec.org/joern/index.shtml', 'http://mlsec.org/sally/api/structfentry__t.html', 'http://mlsec.org/images/bookmark_48.png', 'http://mlsec.org/joern.1', 'http://mlsec.org/images/down_48.png', 'http://mlsec.org/joern/images/down_48.png', 'http://mlsec.org/malheur/api/annotated.html', 'http://mlsec.org/sally/depend.html', 'http://mlsec.org/sally/examples.html', 'http://mlsec.org/images/bag2_48.png', 'http://mlsec.org/harry/api/functions.html', 'http://mlsec.org/images/pencil_48.png', 'http://mlsec.org/joern/images/pencil_48.png', 'http://mlsec.org/joern/download.shtml', 'http://mlsec.org/malheur/api/modules.html', 'http://mlsec.org/salad/docs/salad-stats.html', 'http://mlsec.org/harry/download.html', 'http://mlsec.org/salad/docs.html', 'http://mlsec.org/harry/docs.html', 'http://mlsec.org/images/label_48.png', 'http://mlsec.org/joern/images/label_48.png', 'http://mlsec.org/images/info_48.png', 'http://mlsec.org/joern/images/info_48.png', 'http://mlsec.org/malheur/install.html', 'http://mlsec.org/malheur/api/structindex__t.html', 'http://mlsec.org/sally/install.html', 'http://mlsec.org/salad/docs/salad.pdf', 'http://mlsec.org/salad/install.html', 'http://mlsec.org/salad/install_ubuntu.html', 'http://mlsec.org/harry/install.html', 'http://mlsec.org/malheur/api/structcount__t.html', 'http://mlsec.org/harry/api/structhstring__t.html', 'http://mlsec.org/sally/docs.html', 'http://mlsec.org/harry/api/structdefault__t.html', 'http://mlsec.org/salad/index.html', 'http://mlsec.org/sally/download.html', 'http://mlsec.org/sally/api/structfvec__t.html', 'http://mlsec.org/salad.1', 'http://mlsec.org/sally/api/structconfig__default__t.html', 'http://mlsec.org/malheur/api/structlabel__t.html', 'http://mlsec.org/malheur/docs.html', 'http://mlsec.org/images/info_64.png', 'http://mlsec.org/malheur/download.html', 'http://mlsec.org/malheur/api/structconfig__default__t.html', 'http://mlsec.org/joern/docs.shtml', 'http://mlsec.org/images/bug_48.png', 'http://mlsec.org/salad/install_fromsource.html', 'http://mlsec.org/sally/index.html', 'http://mlsec.org/salad/download.html', 'http://mlsec.org/harry/index.html', 'http://mlsec.org/sally.1', 'http://mlsec.org/harry.1', 'http://mlsec.org/harry/api/group__default.html', 'http://mlsec.org/sally/api/group__sconfig.html', 'http://mlsec.org/malheur/api/structhist__t.html', 'http://mlsec.org/malheur/api/structfentry__t.html', 'http://mlsec.org/harry/api/structhmatrix__t.html', 'http://mlsec.org/malheur/index.html', 'http://mlsec.org/malheur/api/structcluster__t.html', 'http://mlsec.org/salad/docs/manual.html', 'http://mlsec.org/index.html', 'http://mlsec.org/salad/example1.html', 'http://mlsec.org/malheur/examples.html', 'http://mlsec.org/salad/docs/salad-predict.html', 'http://mlsec.org/malheur/api/functions.html', 'http://mlsec.org/malheur/api/group__mist.html', 'http://mlsec.org/sally/example1.html', 'http://mlsec.org/harry/example1.html', 'http://mlsec.org/sally/example2.html', 'http://mlsec.org/malheur/api/structassign__t.html', 'http://mlsec.org/salad/docs/salad-inspect.html', 'http://mlsec.org/salad/docs/salad-train.html', 'http://mlsec.org/harry/example2.html', 'http://mlsec.org/sally/example3.html', 'http://mlsec.org/joern/wallofbugs.shtml', 'http://mlsec.org/joern/default.css', 'http://mlsec.org/harry/api/group__rwlock.html', 'http://mlsec.org/salad/example2.html', 'http://mlsec.org/default.css', 'http://mlsec.org/sally/api/group__reduce.html', 'http://mlsec.org/malheur/images/cws-logo.png', 'http://mlsec.org/harry/api/group__vcache.html', 'http://mlsec.org/harry/examples/alexa/alexa1000.txt', 'http://mlsec.org/malheur/api/structfvec__t.html', 'http://mlsec.org/harry/tutorial.html', 'http://mlsec.org/sally/api/group__fhash.html', 'http://mlsec.org/malheur/api/group__mconfig.html', 'http://mlsec.org/malheur/api/structfarray__t.html', 'http://mlsec.org/malheur/docs/malheur.txt', 'http://mlsec.org/sally/api/group__util.html', 'http://mlsec.org/malheur/api/group__proto.html', 'http://mlsec.org/malheur/changes.html', 'http://mlsec.org/malheur/api/doxygen.css', 'http://mlsec.org/harry/api/doxygen.css', 'http://mlsec.org/sally/api/doxygen.css', 'http://mlsec.org/harry/api/group__util.html', 'http://mlsec.org/sally/changes.html', 'http://mlsec.org/harry/changes.html', 'http://mlsec.org/sally/images/sally2_thm.png', 'http://mlsec.org/images/malheur_logo.png', 'http://mlsec.org/malheur/manual.html', 'http://mlsec.org/sally/images/sally3_thm.png', 'http://mlsec.org/sally/docs/sally.txt', 'http://mlsec.org/sally/images/sally1_thm.png', 'http://mlsec.org/harry/api/group__matrix.html', 'http://mlsec.org/sally/manual.html', 'http://mlsec.org/harry/images/harry_logo.png', 'http://mlsec.org/salad/files/salad-0.3.4.tar.gz', 'http://mlsec.org/sally/api/group__input.html', 'http://mlsec.org/malheur/api/group__class.html', 'http://mlsec.org/harry/api/group__input.html', 'http://mlsec.org/harry/docs/harry.txt', 'http://mlsec.org/harry/api/group__string.html', 'http://mlsec.org/salad/files/salad-0.5.0-1-linux-x86_64.tar.gz', 'http://mlsec.org/salad/images/salad-logo.png', 'http://mlsec.org/malheur/api/group__quality.html', 'http://mlsec.org/salad/files/salad-0.5.0-linux-x86_64.tar.gz', 'http://mlsec.org/sally/images/sally_logo.png', 'http://mlsec.org/salad/files/salad-0.3.2.tar.gz', 'http://mlsec.org/salad/files/salad-0.5.0-1-linux-x86_64.deb', 'http://mlsec.org/salad/files/salad-0.5.0-linux-x86_64.deb', 'http://mlsec.org/sally/api/group__output.html', 'http://mlsec.org/malheur/api/group__ftable.html', 'http://mlsec.org/harry/api/group__output.html', 'http://mlsec.org/salad/files/salad-0.3.5.tar.gz', 'http://mlsec.org/harry/manual.html', 'http://mlsec.org/salad/files/salad-0.5.0-1-linux-x86_64.rpm', 'http://mlsec.org/salad/files/salad-0.5.0-linux-x86_64.rpm', 'http://mlsec.org/malheur/api/group__cluster.html', 'http://mlsec.org/joern/images/joern-logo.png', 'http://mlsec.org/sally/api/group__fvec.html', 'http://mlsec.org/sally/images/sally3.png', 'http://mlsec.org/sally/images/sally1.png', 'http://mlsec.org/malheur/api/group__util.html', 'http://mlsec.org/salad/files/salad-0.4.0.tar.gz', 'http://mlsec.org/harry/api/group__measures.html', 'http://mlsec.org/sally/images/sally2.png', 'http://mlsec.org/salad/files/salad-0.4.3.tar.gz', 'http://mlsec.org/salad/files/salad-0.5.0-1-linux-x86_64.sh', 'http://mlsec.org/salad/files/salad-0.5.0-linux-x86_64.sh', 'http://mlsec.org/harry/examples/alexa/figure_1.png', 'http://mlsec.org/malheur/api/group__fvec.html', 'http://mlsec.org/salad/files/salad-0.6.0-linux-x86_64.tar.gz', 'http://mlsec.org/salad/files/salad-0.6.0-linux-x86_64.deb', 'http://mlsec.org/malheur/api/group__export.html', 'http://mlsec.org/salad/files/salad-0.4.1.tar.gz', 'http://mlsec.org/salad/files/salad-0.4.2.tar.gz', 'http://mlsec.org/malheur/api/jquery.js', 'http://mlsec.org/malheur/api/group__fmath.html', 'http://mlsec.org/salad/files/salad-0.6.0-linux-x86_64.sh', 'http://mlsec.org/malheur/docs/malheur.pdf', 'http://mlsec.org/sally/docs/sally.pdf', 'http://mlsec.org/malheur/api/group__farray.html', 'http://mlsec.org/harry/docs/harry.pdf', 'http://mlsec.org/harry/examples/alexa/figure_2.png', 'http://mlsec.org/harry/api/jquery.js', 'http://mlsec.org/sally/api/jquery.js', 'http://mlsec.org/malheur/docs/mist-tr.pdf', 'http://mlsec.org/salad/examples/ex1-train.zip', 'http://mlsec.org/sally/examples/matrix.png', 'http://mlsec.org/harry/docs/2011-dmkd.pdf', 'http://mlsec.org/salad/examples/ex2-train.zip', 'http://mlsec.org/salad/files/salad-0.5.0.tar.gz', 'http://mlsec.org/salad/files/salad-0.5.0-1.tar.gz', 'http://mlsec.org/salad/files/salad-0.5.0-windows-x86.exe', 'http://mlsec.org/harry/examples/reuters/reuters.zip', 'http://mlsec.org/salad/files/salad-0.6.0-windows-x86.exe', 'http://mlsec.org/salad/examples/ex1-test.zip', 'http://mlsec.org/sally/files/sally-0.4.0.tar.gz', 'http://mlsec.org/sally/files/sally-0.5.0.tar.gz', 'http://mlsec.org/sally/files/sally-0.5.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.5.2.tar.gz', 'http://mlsec.org/malheur/docs/malheur-jcs.pdf', 'http://mlsec.org/sally/docs/2012-jmlr.pdf', 'http://mlsec.org/malheur/files/malheur-0.4.3.tar.gz', 'http://mlsec.org/sally/files/sally-0.6.4.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.4.5.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.4.4.tar.gz', 'http://mlsec.org/sally/files/sally-0.6.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.6.0.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.4.6.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.4.8.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.4.7.tar.gz', 'http://mlsec.org/sally/files/sally-0.6.2.tar.gz', 'http://mlsec.org/sally/files/sally-0.6.3.tar.gz', 'http://mlsec.org/sally/files/sally-0.7.tar.gz', 'http://mlsec.org/sally/files/sally-0.7.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.8.0.tar.gz', 'http://mlsec.org/sally/files/sally-0.8.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.8.2.tar.gz', 'http://mlsec.org/sally/docs/2008-jmlr.pdf', 'http://mlsec.org/malheur/files/malheur-0.5.4.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.5.2.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.5.1.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.5.0.tar.gz', 'http://mlsec.org/malheur/files/malheur-0.5.3.tar.gz', 'http://mlsec.org/sally/examples/dna_hg16.txt.gz', 'http://mlsec.org/harry/files/harry-0.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.8.3.tar.gz', 'http://mlsec.org/sally/files/sally-0.9.0.tar.gz', 'http://mlsec.org/harry/files/harry-0.2.tar.gz', 'http://mlsec.org/sally/files/sally-0.9.1.tar.gz', 'http://mlsec.org/sally/files/sally-0.9.2.tar.gz', 'http://mlsec.org/harry/files/harry-0.3.0.tar.gz', 'http://mlsec.org/sally/files/sally-1.0.0.tar.gz', 'http://mlsec.org/salad/files/salad-0.6.0.tar.gz', 'http://mlsec.org/joern/docs/2014-inbot.pdf', 'http://mlsec.org/salad/docs/2013-aisec.pdf', 'http://mlsec.org/harry/files/harry-0.3.1.tar.gz', 'http://mlsec.org/harry/files/harry-0.3.2.tar.gz', 'http://mlsec.org/harry/files/harry-0.4.0.tar.gz', 'http://mlsec.org/sally/examples/reuters.zip', 'http://mlsec.org/sally/examples/jrc.zip'];
