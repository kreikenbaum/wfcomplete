"use strict";

/**
* @fileoverview stores approximate sizes of HTML files and number of
* embedded elements.
*/

const BloomSort = require("./bloom-sort.js");

// td: as hidden preferences
// use: const Simple = require('sdk/simple-prefs');
const HTML_SIZES = [303, 525, 689, 32034, 242532]; // quantiles (1,3,5,7,9)/10
const HTML_SPLITS = [468, 562, 955, 113337];       // quantiles (2,4,6,8)/10
const NUM_EMBEDDED_SIZES = [4, 10, 20, 42, 82];    // quantiles (1,3,5,7,9)/10
const NUM_EMBEDDED_SPLITS = [6, 13, 27, 54];       // quantiles (2,4,6,8)/10

let htmlSizes = new BloomSort.BloomSort(HTML_SIZES, HTML_SPLITS);
let numEmbeddeds = new BloomSort.BloomSort(NUM_EMBEDDED_SIZES,
                                           NUM_EMBEDDED_SPLITS);

function htmlSize(url) {
    //    console.log('htmlSize(' + url +')');
    return htmlSizes.query(stripParamHref(url));
}
exports.htmlSize = (url) => htmlSize(url);

/**
 * @param {String} url for which to get the size
 *
 * @return {Number} {@code queryMax} of the bin in which url lies.
 */
function htmlSizeMax(url) {
    // console.log('htmlSizeMax(' + url +')');
    return htmlSizes.queryMax(stripParamHref(url));
}
exports.htmlSizeMax = (url) => htmlSizeMax(url);

function numberEmbeddedObjects(url) {
    return numEmbeddeds.query(stripParamHref(url));
}
exports.numberEmbeddedObjects = (url) => numberEmbeddedObjects(url);

function numberEmbeddedObjectsMax(url) {
    return numEmbeddeds.queryMax(stripParamHref(url));
}
exports.numberEmbeddedObjectsMax = (url) => numberEmbeddedObjectsMax(url);

/**
* @param {String} href to strip
*
* @return {String} href without parameters and refs
*/
function stripParamHref(url) {
    //    console.log('stripParamHref(' + url + ')');
    if ( url.indexOf('#') !== -1 ) {
	url = url.slice(0, url.indexOf('#'));
    }
    if ( url.indexOf('?') !== -1 ) {
	url = url.slice(0, url.indexOf('?'));
    }
    if ( url[url.length - 1] === '/' ) {
	url = url.slice(0, -1);
    }
    //    console.log('return: ' + url);
    return url;
}
exports.stripParamHref = url => stripParamHref(url); // testing

function _init() {
    // TESTING
    // const HTML_JSON = [["http://127.0.0.1:1234", 1000], ["https://shavar.services.mozilla.com/downloads", 200]];
    // const NUM_EMBEDDED_JSON = [["http://127.0.0.1:1234/tmptest.html", 30]];

    // REAL
    const HTML_JSON = [["http://google.com", 613], ["http://www.bbc.co.uk", 141057], ["http://stackoverflow.com", 244915], ["http://www.booking.com", 342274], ["http://craigslist.org", 386], ["http://t.co", 3525], ["http://alipay.com", 578], ["http://passport.weibo.com/visitor/visitor", 6180], ["http://fc2.com", 35706], ["http://www.google.de", 11492], ["http://walmart.com", 677], ["http://www.dailymotion.com/de", 146384], ["http://netflix.com", 922], ["http://soso.com", 526], ["http://www.nicovideo.jp", 106511], ["http://bongacams.com", 782], ["http://www.ettoday.net", 237762], ["http://www.walmart.com", 419441], ["http://bankofamerica.com", 276], ["http://www.wikia.com/fandom", 214975], ["http://popads.net", 920], ["http://www.msn.com", 2109], ["http://linkedin.com", 278], ["http://www.qq.com", 625707], ["http://instagram.com", 302], ["http://www.nytimes.com", 178624], ["http://weibo.com", 725], ["http://adcash.com", 525], ["http://www.uol.com.br", 360573], ["http://ameblo.jp", 38322], ["http://www.cntv.cn", 590], ["http://microsoftonline.com", 558], ["http://www.sogou.com", 17929], ["http://chase.com", 269], ["http://go.com", 119616], ["http://www.hao123.com", 732126], ["http://hao123.com", 271], ["http://www.dailymotion.com", 1032], ["http://login.microsoftonline.com", 534], ["http://www.cnn.com", 431], ["http://twitter.com", 567], ["http://www.live.com", 776], ["http://mail.ru", 740], ["http://ettoday.net", 270], ["http://baidu.com", 492], ["http://www.xvideos.com", 48226], ["http://wikipedia.org", 570], ["http://livedoor.jp", 574], ["http://github.com", 251], ["http://espn.go.com", 322089], ["http://www.microsoft.com", 305], ["http://coccoc.com", 42487], ["http://www.blogger.com", 879], ["http://china.com", 244576], ["http://www.msn.com/de-de", 39110], ["http://imdb.com", 562], ["http://live.com", 410], ["http://www.livedoor.com", 63123], ["http://aliexpress.com", 717], ["http://pixnet.net", 442], ["http://onclickads.net", 550], ["http://xhamster.com", 54466], ["http://alibaba.com", 686], ["http://cntv.cn", 523], ["http://dropbox.com", 528], ["http://taobao.com", 556], ["http://imgur.com", 153910], ["http://www.sina.com.cn", 547301], ["http://xnxx.com", 542], ["http://www.ask.com", 937], ["http://www.amazon.com", 315034], ["http://www.163.com", 751070], ["http://sohu.com", 511], ["http://www.theguardian.com", 632], ["http://www.bankofamerica.com", 277], ["http://www.tianya.cn", 11263], ["http://amazon.com", 547], ["http://www.directrev.com", 7304], ["http://www.sogou.com", 708], ["http://bbc.co.uk", 278], ["http://www.ebay.com", 170613], ["http://naver.com", 507], ["http://dmm.co.jp", 385], ["http://www.360.com", 87560], ["http://www.wikia.com", 1082], ["http://sogou.com", 473], ["http://nicovideo.jp", 551], ["http://www.sohu.com", 424118], ["http://www.xnxx.com", 82983], ["http://www.huffingtonpost.de", 220038], ["http://frankfurt.craigslist.de", 37977], ["http://www.apple.com", 31626], ["http://adobe.com", 545], ["http://pinterest.com", 608], ["http://www.flipkart.com", 69922], ["http://jd.com", 468], ["http://www.microsoft.com/de-de", 80759], ["http://nytimes.com", 468], ["http://www.jd.com", 191821], ["http://sina.com.cn", 594], ["http://theguardian.com", 427], ["http://blogspot.com", 690], ["http://vk.com", 18164], ["http://gmw.cn", 141149], ["http://reddit.com", 499], ["http://directrev.com", 401], ["http://www.alibaba.com", 118198], ["http://tmall.com", 555], ["http://dailymotion.com", 284], ["http://msn.com", 485], ["http://www.taobao.com", 688], ["http://geo.craigslist.org", 418], ["http://pornhub.com", 255], ["http://chinadaily.com.cn", 481], ["http://www.naver.com", 85546], ["http://ask.com", 689], ["http://aws.amazon.com", 338894], ["http://amazonaws.com", 564], ["http://indiatimes.com", 555], ["http://cnn.com", 552], ["http://uol.com.br", 523], ["http://www.yandex.ru", 556], ["http://www.rakuten.co.jp", 339756], ["http://163.com", 471], ["http://www.outbrain.com", 20769], ["http://bing.com", 410], ["http://edition.cnn.com", 110097], ["http://flipkart.com", 476], ["http://facebook.com", 403], ["http://xvideos.com", 548], ["http://ebay.com", 389], ["http://dailymail.co.uk", 451], ["http://www.weibo.com", 804], ["http://www.tmall.com", 678], ["http://www.dailymail.co.uk", 313], ["http://tianya.cn", 277], ["http://www.popads.net", 763], ["http://www.chinadaily.com.cn", 10959], ["http://booking.com", 317], ["http://www.dmm.co.jp", 27799], ["http://www.aliexpress.com", 50293], ["http://www.adobe.com", 130211], ["http://www.dailymail.co.uk/home/index.html", 888887], ["http://paypal.com", 266], ["http://tumblr.com", 270], ["http://youtube.com", 335], ["http://www.craigslist.org", 272], ["http://blogger.com", 690], ["http://www.imdb.com", 131393], ["http://outbrain.com", 256], ["http://apple.com", 284], ["http://kat.cr", 467], ["http://www.bing.com", 85061], ["http://www.pornhub.com", 220656], ["http://buzzfeed.com", 281], ["http://xinhuanet.com", 248513], ["http://www.buzzfeed.com", 568226], ["http://www.indiatimes.com", 167865], ["http://www.theguardian.com/international", 562657], ["http://www.onclickads.net", 3273], ["http://de.ask.com", 22713], ["http://yandex.ru", 663], ["http://diply.com", 15490], ["http://qq.com", 600], ["http://yahoo.com", 809], ["http://wikia.com", 959], ["http://360.cn", 508], ["http://www.huffingtonpost.com", 525], ["http://www.youtube.com", 816], ["http://huffingtonpost.com", 563], ["http://wordpress.com", 487], ["http://microsoft.com", 497], ["http://ok.ru", 125422], ["http://rakuten.co.jp", 553]];
    const NUM_EMBEDDED_JSON = [["http://www.bbc.co.uk", 30], ["https://www.chase.com", 13], ["https://www.bankofamerica.com", 18], ["https://www.google.de", 4], ["https://www.reddit.com", 31], ["http://gmw.cn", 62], ["http://www.dailymotion.com/de", 36], ["http://www.outbrain.com", 26], ["https://www.tmall.com", 7], ["http://www.nicovideo.jp", 72], ["https://www.linkedin.com", 10], ["http://www.ettoday.net", 103], ["http://diply.com", 26], ["http://www.walmart.com", 44], ["http://www.wikia.com/fandom", 41], ["https://accounts.google.com/ServiceLogin", 2], ["http://www.qq.com", 6], ["http://ameblo.jp", 21], ["http://www.nytimes.com", 54], ["http://www.uol.com.br", 92], ["https://www.hao123.com", 4], ["http://stackoverflow.com", 9], ["http://vk.com", 15], ["https://mail.ru", 42], ["https://www.popads.net", 12], ["http://t.co", 2], ["https://www.paypal.com/de/webapps/mpp/home", 26], ["https://www.taobao.com", 8], ["http://www.aliexpress.com:80", 21], ["https://de.wordpress.com", 5], ["http://www.xvideos.com", 8], ["https://www.instagram.com", 3], ["http://go.com", 24], ["http://www.msn.com/de-de", 13], ["http://www.livedoor.com", 31], ["http://passport.weibo.com/visitor/visitor", 1], ["http://de.ask.com", 6], ["http://www.sina.com.cn", 71], ["https://www.netflix.com/de", 6], ["http://www.amazon.com", 138], ["http://www.tianya.cn", 10], ["https://de.bongacams.com", 6], ["http://edition.cnn.com", 22], ["http://www.directrev.com", 19], ["http://www.sogou.com", 9], ["http://www.booking.com/index.en-gb.html", 47], ["http://www.ebay.com", 14], ["http://www.360.com", 144], ["http://imgur.com", 79], ["http://www.sohu.com", 61], ["http://www.xnxx.com", 58], ["http://www.huffingtonpost.de", 49], ["http://frankfurt.craigslist.de", 5], ["http://www.apple.com", 11], ["http://www.flipkart.com", 26], ["http://www.microsoft.com/de-de", 38], ["https://www.facebook.com", 11], ["https://www.wikipedia.org", 6], ["http://xinhuanet.com", 33], ["https://www.youtube.com", 54], ["http://baidu.com", 0], ["https://www.adcash.com/en/index.php", 43], ["http://www.naver.com", 31], ["https://www.pinterest.com", 19], ["http://coccoc.com", 19], ["https://de.yahoo.com", 26], ["http://www.163.com", 20], ["https://twitter.com", 59], ["http://www.jd.com", 5], ["http://www.rakuten.co.jp", 159], ["https://login.microsoftonline.com:443", 6], ["http://espn.go.com", 10], ["https://login.live.com/login.srf", 4], ["http://fc2.com", 20], ["https://www.pixnet.net", 114], ["https://www.sogou.com", 10], ["http://www.alibaba.com", 10], ["http://aws.amazon.com", 123], ["http://www.chinadaily.com.cn", 44], ["http://www.dmm.co.jp", 14], ["http://china.com", 212], ["https://www.alipay.com", 2], ["http://www.adobe.com", 15], ["http://www.dailymail.co.uk/home/index.html", 556], ["http://www.imdb.com", 60], ["http://ok.ru", 44], ["http://www.bing.com", 3], ["https://github.com", 11], ["http://www.pornhub.com", 41], ["https://kat.cr", 0], ["https://www.yandex.ru", 10], ["http://www.buzzfeed.com", 79], ["http://www.indiatimes.com", 108], ["http://www.theguardian.com/international", 11], ["http://www.onclickads.net", 6], ["https://www.tumblr.com", 27], ["http://www.cntv.cn", 0], ["http://xhamster.com", 55], ["https://www.dropbox.com", 42]];
    for (let i = 0, l=HTML_JSON.length; i < l; i++) {
        htmlSizes.add(HTML_JSON[i][0], HTML_JSON[i][1]);
    }
    for (let i = 0, l=NUM_EMBEDDED_JSON.length; i < l; i++) {
        numEmbeddeds.add(NUM_EMBEDDED_JSON[i][0], NUM_EMBEDDED_JSON[i][1]);
    }
}

_init();
