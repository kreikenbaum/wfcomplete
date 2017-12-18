#!/usr/bin/env python
'''starts tor-firefox -marionette and tshark, calls url parameter, stops both'''

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
from marionette_driver.marionette import Marionette

import config

ERRFILENAME="/tmp/one_site_err.txt"

def browse_to(page, bridge=None):
    '''creates a browser instance, packet dump, browses to page, kills both.
    If bridge is not none, it is an IP-address. Just capture traffic to that.'''
    browser = _open_browser()
    _open_with_timeout(browser, page, bridge=bridge)

# cheap hack
def _avoid_safe_mode(exedir):
    '''avoids safe mode by removing the line which contains count of failures'''
    os.system(r"sed -i '/toolkit\.startup\.recent_crashes/d' " +
              os.path.join(exedir,
                           "TorBrowser/Data/Browser/profile.default/prefs.js"))


def _kill(*processes):
    '''cleans up after processes'''
    for process in processes:
        if process is not None:
            process.terminate()


def _navigate_or_fail(client, url, file_name):
    '''navigates client to url, on failure renames file'''
    try:
        client.navigate(url)
    except:
        with open(ERRFILENAME, "a") as f:
            f.write('url: ' + url + "\n")
            f.write('file: ' + file_name + "\n")
            f.write('dir: ' + os.getcwd() + "\n\n")
            f.write(str(sys.exc_info()))
            f.write('\n')
        try:
            os.rename(
                file_name,
                '{}_{}'.format(
                    file_name,
                    sys.exc_info()[1])
                .replace(' ', '_').replace('\n', ''))
        except IOError:
            print 'failed with IOError'
            print file_name
            print sys.exc_info()


def _normalize_url(url='google.com'):
    '''normalizes the url by adding the "http://' scheme if none exists'''
    if not urlparse.urlparse(url).scheme:
        url = "http://" + url

    return (url, urlparse.urlparse(url).netloc)


def _open_browser(exe='/home/mkreik/bin/tor-browser_en-US/Browser/firefox -marionette', open_timeout=60):
    '''returns an initialized browser instance with marionette'''
    env_with_debug = os.environ.copy()
    env_with_debug["MOZ_DISABLE_AUTO_SAFE_MODE"] = 'set'
    exewholepath, exeargs = exe.split(' ', 1)
    (exedir, _) = os.path.split(exewholepath)
    env_with_debug["LD_LIBRARY_PATH"] = '/lib:/usr/lib:' + (
        exedir + '/TorBrowser/Tor')
    _avoid_safe_mode(exedir)
    #print 'lpath: %s' % env_with_debug["LD_LIBRARY_PATH"]
# maybe remove lines below
#    browser =
#    subprocess.Popen(args=['/home/mkreik/bin/tor-browser_en-US/Browser/firefox','-marionette'],
#    cwd=exedir, stdout=subprocess.PIPE, env=env_with_debug);
    browser = subprocess.Popen(
        args=[exewholepath, exeargs], cwd=exedir, stdout=subprocess.PIPE,
        stderr=open(os.path.join(config.SAVETO, ERRFILENAME), "a"),
        env=env_with_debug)

    thread = threading.Thread(target=_wait_browser_ready,
                              kwargs={'browser': browser})
    thread.daemon = True
    thread.start()

    start = time.time()
    while thread.is_alive():
        time.sleep(.1)
        if time.time() - start > open_timeout:
            _kill(browser)
            raise SystemExit("browser connection not working")
    print 'slept for {0:.3f} seconds'.format(time.time() - start)
    return browser


def _open_packet_dump(page, bridge):
    '''@returns a (tshark_subprocess_instance, filename) tuple'''
    loc = os.path.join(config.SAVETO,
                       page + '@' + str(time.time()).split('.')[0])
    if not bridge:
        return (subprocess.Popen(['tshark', '-w' + loc]), loc)
    else:
        return (subprocess.Popen(['tshark', '-w' + loc, 'host ' + bridge]), loc)


def _open_with_timeout(browser, page, timeout=600, burst_wait=3, bridge=None):
    '''navigates browser to url while capturing the packet dump, aborts
    after timeout. If bridge, that is the IP address of the connected
    bridge, just capture traffic to there (need to set this by hand)

    '''
    client = Marionette('localhost', port=2828, socket_timeout=(timeout-30))
    client.start_session()

    (url, domain) = _normalize_url(page)

    (tshark_process, file_name) = _open_packet_dump(domain, bridge)
#    thread = threading.Thread(target=client.navigate, args=(url,))
    thread = threading.Thread(target=_navigate_or_fail,
                              args=(client, url, file_name))
    thread.daemon = True
    thread.start()

    #td: this is code duplication for both _open functions
    start = time.time()
    while thread.is_alive():
        time.sleep(.1)
        if time.time() - start > timeout:
            _kill(browser, tshark_process)
            os.rename(file_name, file_name + '_timeout')
            raise SystemExit("download aborted after timeout")
    time.sleep(burst_wait)
    _kill(browser, tshark_process)


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


if __name__ == "__main__":
    if len(sys.argv) == 2:
        browse_to(sys.argv[1])
    else:
        browse_to(sys.argv[1], sys.argv[2])
