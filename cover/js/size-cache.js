"use strict";

const BloomSort = require("./bloom-sort.js");
const stats = require("./stats.js");

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
    const json = '[["http://google.com/", 609], ["http://www.google.de/?", 20354], ["http://facebook.com/", 403], ["http://youtube.com/", 335], ["http://www.youtube.com/", 816], ["http://baidu.com/", 492], ["http://yahoo.com/", 809], ["http://amazon.com/", 1799], ["http://wikipedia.org/", 571], ["http://qq.com/", 600], ["http://www.qq.com/", 630965], ["http://twitter.com/", 567], ["http://taobao.com/", 556], ["http://www.taobao.com/", 688], ["http://live.com/", 410], ["http://www.live.com/", 775], ["http://sina.com.cn/", 594], ["http://www.sina.com.cn/", 531421], ["http://linkedin.com/", 278], ["http://weibo.com/", 725], ["http://www.weibo.com/", 806], ["http://passport.weibo.com/visitor/visitor?", 6173], ["http://ebay.com/", 392], ["http://www.ebay.com/", 171832], ["http://yandex.ru/", 319], ["http://www.yandex.ru/", 665], ["http://vk.com/", 18024], ["http://hao123.com/", 271], ["http://www.hao123.com/", 703374], ["http://bing.com/", 410], ["http://www.bing.com/", 83680], ["http://t.co/", 3525], ["http://msn.com/", 485], ["http://www.msn.com/", 2109], ["http://www.msn.com/de-de/", 40501], ["http://instagram.com/", 302], ["http://aliexpress.com/", 716], ["http://www.aliexpress.com/", 48326], ["http://blogspot.com/", 690], ["http://www.blogger.com/", 879], ["http://apple.com/", 284], ["http://www.apple.com/", 26887], ["http://ask.com/", 689], ["http://www.ask.com/", 937], ["http://de.ask.com/?", 19288], ["http://pinterest.com/", 608], ["http://wordpress.com/", 487], ["http://tmall.com/", 555], ["http://www.tmall.com/", 601], ["http://reddit.com/", 499], ["http://mail.ru/", 740], ["http://paypal.com/", 3826], ["http://onclickads.net/", 549], ["http://www.onclickads.net/", 3273], ["http://sohu.com/", 511], ["http://www.sohu.com/", 423747], ["http://tumblr.com/", 270], ["http://imgur.com/", 153921], ["http://microsoft.com/", 497], ["http://www.microsoft.com/", 305], ["http://www.microsoft.com/de-de/", 80157], ["http://gmw.cn/", 155062], ["http://xvideos.com/", 548], ["http://www.xvideos.com/", 47784], ["http://imdb.com/", 562], ["http://www.imdb.com/", 123153], ["http://fc2.com/", 35706], ["http://netflix.com/", 922], ["http://googleadservices.com/", 1896], ["http://360.cn/", 508], ["http://www.360.com/", 113633], ["http://stackoverflow.com/", 245748], ["http://go.com/", 121680], ["http://alibaba.com/", 686], ["http://www.alibaba.com/", 117665], ["http://ok.ru/", 132889], ["http://craigslist.org/", 386], ["http://www.craigslist.org/", 272], ["http://geo.craigslist.org/", 417], ["http://hannover.craigslist.de/", 37592], ["http://tianya.cn/", 277], ["http://www.tianya.cn/", 11039], ["http://rakuten.co.jp/", 553], ["http://www.rakuten.co.jp/", 297014], ["http://pornhub.com/", 255], ["http://www.pornhub.com/", 217571], ["http://blogger.com/", 690], ["http://www.blogger.com/", 879], ["http://naver.com/", 507], ["http://www.naver.com/", 86867], ["http://espn.go.com/", 344671], ["http://xhamster.com/", 55004], ["http://outbrain.com/", 256], ["http://www.outbrain.com/", 20985], ["http://cnn.com/", 552], ["http://www.cnn.com/", 386], ["http://edition.cnn.com/", 110517], ["http://soso.com/", 526], ["http://www.sogou.com/?", 16498], ["http://kat.cr/", 467], ["http://nicovideo.jp/", 551], ["http://www.nicovideo.jp/", 104611], ["http://xinhuanet.com/", 264441], ["http://bbc.co.uk/", 278], ["http://www.bbc.co.uk/", 167807], ["http://diply.com/", 15490], ["http://flipkart.com/", 476], ["http://www.flipkart.com/", 65804], ["http://github.com/", 251], ["http://dropbox.com/", 528], ["http://googleusercontent.com/", 3870], ["http://adcash.com/", 525], ["http://popads.net/", 920], ["http://www.popads.net/", 763], ["http://cntv.cn/", 523], ["http://www.cntv.cn/", 590], ["http://pixnet.net/", 442], ["http://dailymotion.com/", 284], ["http://www.dailymotion.com/", 1036], ["http://www.dailymotion.com/de", 139079], ["http://jd.com/", 468], ["http://www.jd.com/", 200676], ["http://booking.com/", 317], ["http://www.booking.com/", 331566], ["http://163.com/", 471], ["http://www.163.com/", 751801], ["http://nytimes.com/", 497], ["http://www.nytimes.com/", 185354], ["http://sogou.com/", 473], ["http://www.sogou.com/", 708], ["http://china.com/", 244870], ["http://livedoor.jp/", 574], ["http://www.livedoor.com/", 62967], ["http://dailymail.co.uk/", 451], ["http://www.dailymail.co.uk/", 313], ["http://www.dailymail.co.uk/home/index.html", 829208], ["http://wikia.com/", 959], ["http://www.wikia.com/", 1084], ["http://www.wikia.com/fandom", 206713], ["http://indiatimes.com/", 555], ["http://www.indiatimes.com/", 168798], ["http://adobe.com/", 545], ["http://www.adobe.com/", 130434], ["http://uol.com.br/", 523], ["http://www.uol.com.br/", 342232], ["http://alipay.com/", 578], ["http://chase.com/", 269], ["http://huffingtonpost.com/", 563], ["http://www.huffingtonpost.com/", 526], ["http://www.huffingtonpost.de/", 228831], ["http://coccoc.com/", 41888], ["http://buzzfeed.com/", 281], ["http://www.buzzfeed.com/", 562900], ["http://xnxx.com/", 542], ["http://www.xnxx.com/", 83405], ["http://youku.com/", 310], ["http://www.youku.com/", 242186], ["http://www.youku.com/", 390504], ["http://adnetworkperformance.com/", 293], ["http://dmm.co.jp/", 385], ["http://www.dmm.co.jp/", 3838], ["http://directrev.com/", 3838], ["http://bongacams.com/", 895], ["http://ameblo.jp/", 38167], ["http://theguardian.com/", 427], ["http://www.theguardian.com/", 632], ["http://www.theguardian.com/international", 563772], ["http://chinadaily.com.cn/", 481], ["http://www.chinadaily.com.cn/", 10908], ["http://bankofamerica.com/", 276], ["http://www.bankofamerica.com/", 3870]]'
    for (var i = 0, l=json.length; i < l; i++) {
        htmlSizes.add(json[i][0], json[i][1]);
    }
}
