#!/usr/bin/env python

# starts tor-firefox -marionette and tshark, calls url parameter, stops both

# some help was
# http://stackoverflow.com/questions/6344993/problems-parsing-an-url-with-python
import os
import os.path
import socket
import subprocess
import sys
import threading
import time
import urlparse
from marionette import Marionette

def browse_to(page):
    '''creates a browser instance, packet dump, browses to page, kills both'''
    browser = _open_browser()
    _open_with_timeout(browser, page)

# cheap hack
def _avoid_safe_mode(exedir):
    '''avoids safe mode by removing the line which contains count of failures'''
    os.system("sed -i '/toolkit\.startup\.recent_crashes/d' " +
              os.path.join(exedir,
                           "TorBrowser/Data/Browser/profile.default/prefs.js"))

def _open_with_timeout(browser, page, timeout=600, burst_wait=3):
    '''navigates browser to url while capturing the packet dump, aborts after timeout'''
    client = Marionette('localhost', port=2828, socket_timeout=(timeout-30))
    client.start_session()
    client.timeouts('page load', timeout * 1000) # not working

    (url, domain) = _normalize_url(page)

    thread = threading.Thread(target=client.navigate, args=(url,))
    thread.daemon = True
    (pcap, file_name) = _open_packet_dump(domain)
    thread.start()

    #td: this is code duplication for both _open functions
    start = time.time()
    while thread.is_alive():
        time.sleep(.1)
        if time.time() - start > timeout:
            _clean_up(browser, pcap)
            os.rename(file_name, file_name+'timeout')
            raise SystemExit("download aborted after timeout")
    time.sleep(burst_wait)
    _clean_up(browser, pcap)


def _open_browser(exe='/home/mkreik/bin/tor-browser_en-US/Browser/firefox -marionette', tryThisLong = 60):
    '''returns an initialized browser instance with marionette'''
    env_with_debug = os.environ.copy()
    env_with_debug["MOZ_DISABLE_AUTO_SAFE_MODE"] = 'set';
    exewholepath, exeargs = exe.split(' ', 1)
    (exedir, exefile) = os.path.split(exewholepath)
    env_with_debug["LD_LIBRARY_PATH"] = '/lib:/usr/lib:' + (
        exedir + '/TorBrowser/Tor')
    _avoid_safe_mode(exedir)
    #print 'lpath: %s' % env_with_debug["LD_LIBRARY_PATH"]
#    browser = subprocess.Popen(args=['/home/mkreik/bin/tor-browser_en-US/Browser/firefox','-marionette'], cwd=exedir, stdout=subprocess.PIPE, env=env_with_debug);
    browser = subprocess.Popen(args=[exewholepath,exeargs], cwd=exedir, stdout=subprocess.PIPE, env=env_with_debug);

    thread = threading.Thread(target=_wait_browser_ready,
                              kwargs={'browser': browser});
    thread.daemon = True
    thread.start()

    start = time.time()
    while thread.is_alive():
        time.sleep(.1)
        if time.time() - start > tryThisLong:
            _clean_up(browser)
            raise SystemExit("browser connection not working")
    print 'slept for {0:.3f} seconds'.format(time.time() - start)
    return browser

def _wait_browser_ready(browser):
    '''waits for browser to become ready'''
    for line in iter(browser.stdout.readline, ''):
        if 'Bootstrapped 100%: Done' in line:
            break
    client = Marionette('localhost', port=2828)
    connected = False
    while not connected:
        try:
            client.start_session()
            connected = True
        except socket.error:
            time.sleep(.1)
    client.close()

def _clean_up(*processes):
    '''cleans up after processes'''
    for p in processes:
        if p is not None:
            p.terminate()

def _normalize_url(url='google.com'):
    '''normalizes the url by adding the "http://' scheme if none exists'''
    if not urlparse.urlparse(url).scheme:
        url = "http://" + url

    return (url, urlparse.urlparse(url).netloc)

def _open_packet_dump(page):
    '''returns a tshark instance which writes to /mnt/data/'page' '''
    loc = os.path.join('/mnt/data', page + '@' + str(time.time()).split('.')[0]);
    return (subprocess.Popen(['tshark', '-w' + loc, 'host 134.169.109.51']), loc);

if __name__ == "__main__":
    browse_to(sys.argv[1])
