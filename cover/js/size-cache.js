"use strict";

const BloomSort = require("./bloom-sort.js");
const stats = require("./stats.js");

const NUM_EMBEDDED_SIZES = [Math.round(490.73), Math.round(2704.62892304),
                    Math.round(14906.34)]; //quantiles 1/6 3/6 5/6
const NUM_EMBEDDED_SPLITS = [Math.round(1264.95), Math.round(5782.85)]; // quant.1/3,2/3

let htmlSizes = new BloomSort.BloomSort(HTML_SIZES, HTML_SPLITS);

const HTML_SIZES = [Math.round(490.73), Math.round(2704.62892304), 
                    Math.round(14906.34)]; //quantiles 1/6 3/6 5/6
const HTML_SPLITS = [Math.round(1264.95), Math.round(5782.85)]; // quant.1/3,2/3

let htmlSizes = new BloomSort.BloomSort(HTML_SIZES, HTML_SPLITS);


function htmlSize(url) {
    try {
        return htmlSizes.query(url);
    } catch (e) {
        return stats.htmlSize();
    }
}
exports.htmlSize = htmlSize;

function numEmbedded(url) {
    // pass
}

function _populateHtml() {
    // pass
}

function _init() {
    const HTML_JSON = '[["http://google.com/", 613], ["http://www.bbc.co.uk/", 141057], ["http://stackoverflow.com/", 244915], ["http://www.booking.com/", 342274], ["http://craigslist.org/", 386], ["http://t.co/", 3525], ["http://alipay.com/", 578], ["http://passport.weibo.com/visitor/visitor?", 6180], ["http://fc2.com/", 35706], ["http://www.google.de/?", 11492], ["http://walmart.com/", 677], ["http://www.dailymotion.com/de", 146384], ["http://netflix.com/", 922], ["http://soso.com/", 526], ["http://www.nicovideo.jp/", 106511], ["http://bongacams.com/", 782], ["http://www.ettoday.net/", 237762], ["http://www.walmart.com/", 419441], ["http://bankofamerica.com/", 276], ["http://www.wikia.com/fandom", 214975], ["http://popads.net/", 920], ["http://www.msn.com/", 2109], ["http://linkedin.com/", 278], ["http://www.qq.com/", 625707], ["http://instagram.com/", 302], ["http://www.nytimes.com/", 178624], ["http://weibo.com/", 725], ["http://adcash.com/", 525], ["http://www.uol.com.br/", 360573], ["http://ameblo.jp/", 38322], ["http://www.cntv.cn/", 590], ["http://microsoftonline.com/", 558], ["http://www.sogou.com/?", 17929], ["http://chase.com/", 269], ["http://go.com/", 119616], ["http://www.hao123.com/", 732126], ["http://hao123.com/", 271], ["http://www.dailymotion.com/", 1032], ["http://login.microsoftonline.com/", 534], ["http://www.cnn.com/", 431], ["http://twitter.com/", 567], ["http://www.live.com/", 776], ["http://mail.ru/", 740], ["http://ettoday.net/", 270], ["http://baidu.com/", 492], ["http://www.xvideos.com/", 48226], ["http://wikipedia.org/", 570], ["http://livedoor.jp/", 574], ["http://github.com/", 251], ["http://espn.go.com/", 322089], ["http://www.microsoft.com/", 305], ["http://coccoc.com/", 42487], ["http://www.blogger.com/", 879], ["http://china.com/", 244576], ["http://www.msn.com/de-de/", 39110], ["http://imdb.com/", 562], ["http://live.com/", 410], ["http://www.livedoor.com/", 63123], ["http://aliexpress.com/", 717], ["http://pixnet.net/", 442], ["http://onclickads.net/", 550], ["http://xhamster.com/", 54466], ["http://alibaba.com/", 686], ["http://cntv.cn/", 523], ["http://dropbox.com/", 528], ["http://taobao.com/", 556], ["http://imgur.com/", 153910], ["http://www.sina.com.cn/", 547301], ["http://xnxx.com/", 542], ["http://www.ask.com/", 937], ["http://www.amazon.com/", 315034], ["http://www.163.com/", 751070], ["http://sohu.com/", 511], ["http://www.theguardian.com/", 632], ["http://www.bankofamerica.com/", 277], ["http://www.tianya.cn/", 11263], ["http://amazon.com/", 547], ["http://www.directrev.com/", 7304], ["http://www.sogou.com/", 708], ["http://bbc.co.uk/", 278], ["http://www.ebay.com/", 170613], ["http://naver.com/", 507], ["http://dmm.co.jp/", 385], ["http://www.360.com/", 87560], ["http://www.wikia.com/", 1082], ["http://sogou.com/", 473], ["http://nicovideo.jp/", 551], ["http://www.sohu.com/", 424118], ["http://www.xnxx.com/", 82983], ["http://www.huffingtonpost.de/", 220038], ["http://frankfurt.craigslist.de/", 37977], ["http://www.apple.com/", 31626], ["http://adobe.com/", 545], ["http://pinterest.com/", 608], ["http://www.flipkart.com/", 69922], ["http://jd.com/", 468], ["http://www.microsoft.com/de-de/", 80759], ["http://nytimes.com/", 468], ["http://www.jd.com/", 191821], ["http://sina.com.cn/", 594], ["http://theguardian.com/", 427], ["http://blogspot.com/", 690], ["http://vk.com/", 18164], ["http://gmw.cn/", 141149], ["http://reddit.com/", 499], ["http://directrev.com/", 401], ["http://www.alibaba.com/", 118198], ["http://tmall.com/", 555], ["http://dailymotion.com/", 284], ["http://msn.com/", 485], ["http://www.taobao.com/", 688], ["http://geo.craigslist.org/", 418], ["http://pornhub.com/", 255], ["http://chinadaily.com.cn/", 481], ["http://www.naver.com/", 85546], ["http://ask.com/", 689], ["http://aws.amazon.com/", 338894], ["http://amazonaws.com/", 564], ["http://indiatimes.com/", 555], ["http://cnn.com/", 552], ["http://uol.com.br/", 523], ["http://www.yandex.ru/", 556], ["http://www.rakuten.co.jp/", 339756], ["http://163.com/", 471], ["http://www.outbrain.com/", 20769], ["http://bing.com/", 410], ["http://edition.cnn.com/", 110097], ["http://flipkart.com/", 476], ["http://facebook.com/", 403], ["http://xvideos.com/", 548], ["http://ebay.com/", 389], ["http://dailymail.co.uk/", 451], ["http://www.weibo.com/", 804], ["http://www.tmall.com/", 678], ["http://www.dailymail.co.uk/", 313], ["http://tianya.cn/", 277], ["http://www.popads.net/", 763], ["http://www.chinadaily.com.cn/", 10959], ["http://booking.com/", 317], ["http://www.dmm.co.jp/", 27799], ["http://www.aliexpress.com/", 50293], ["http://www.adobe.com/", 130211], ["http://www.dailymail.co.uk/home/index.html", 888887], ["http://paypal.com/", 266], ["http://tumblr.com/", 270], ["http://youtube.com/", 335], ["http://www.craigslist.org/", 272], ["http://blogger.com/", 690], ["http://www.imdb.com/", 131393], ["http://outbrain.com/", 256], ["http://apple.com/", 284], ["http://kat.cr/", 467], ["http://www.bing.com/", 85061], ["http://www.pornhub.com/", 220656], ["http://buzzfeed.com/", 281], ["http://xinhuanet.com/", 248513], ["http://www.buzzfeed.com/", 568226], ["http://www.indiatimes.com/", 167865], ["http://www.theguardian.com/international", 562657], ["http://www.onclickads.net/", 3273], ["http://de.ask.com/?", 22713], ["http://yandex.ru/", 663], ["http://diply.com/", 15490], ["http://qq.com/", 600], ["http://yahoo.com/", 809], ["http://wikia.com/", 959], ["http://360.cn/", 508], ["http://www.huffingtonpost.com/", 525], ["http://www.youtube.com/", 816], ["http://huffingtonpost.com/", 563], ["http://wordpress.com/", 487], ["http://microsoft.com/", 497], ["http://ok.ru/", 125422], ["http://rakuten.co.jp/", 553]]'
    for (var i = 0, l=HTML_JSON.length; i < l; i++) {
        htmlSizes.add(HTML_JSON[i][0], HTML_JSON[i][1]);
    }
}
