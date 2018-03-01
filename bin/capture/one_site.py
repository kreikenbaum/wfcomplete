#!/usr/bin/env python
'''starts tor-browser -marionette, tests connection, starts tshark,
loads url, finishes, stops both'''
import collections
import logging
import os
import psutil
import socket
import subprocess
import sys
import threading
import time
import urlparse

import marionette_driver
from marionette_driver.marionette import Marionette
from marionette_driver import errors

import config
import logconfig
ERRFILENAME = os.path.join('/tmp', "one_site_err.txt")
logconfig.add_file_output(ERRFILENAME)
FIREFOX_PATH = os.path.join(os.getenv("HOME"), 'bin', 'tor-browser_en-US',
                            'Browser', 'firefox')

PROBLEMATIC_DELAY = [
    "test tor network settings",
    "loading ..."  # seen one time at pornhub.com
]
PROBLEMATIC_TEXT = [
    "test tor network settings"
    "cloudflare ray id:",
    "reference #18",
    "you don't have permission to access",
    "access denied",
    "this webpage was generated by the domain owner using sedo domain parking",
    "bad gateway",
    "generated by cloudfront",
    "ddos protection by cloudflare",
    "buy this domain",
    "your browser will redirect to your requested content shortly",
    "verify you're a human to continue",
    "captcha",  # deprecated: could be contained otherwise (but worked well)
    "403 forbidden"  # also deprecated: could be contained otherwise
]


# cheap hack
def _avoid_safe_mode(exedir):
    '''Avoids safe mode.
    Removes the line which contains count of failures.'''
    os.system(r"sed -i '/toolkit\.startup\.recent_crashes/d' " +
              os.path.join(exedir, "TorBrowser", "Data", "Browser",
                           "profile.default", "prefs.js"))


def browse_to(page, bridge=None):
    '''creates a browser instance, packet dump, browses to page, kills both.
    If bridge is not none, it is an IP-address. Capture traffic to that.'''
    browser = _open_browser()
    _open_with_timeout(browser, page, bridge=bridge)


def _check_text(text, file_name=None, client=None):
    '''@return False if problem, and should stop, True if all's well'''
    text = text.lower()
    if not text:
        raise DelayError("empty body")
    for delay_text in PROBLEMATIC_DELAY:
        if delay_text in text:
            raise DelayError(delay_text)
    for problem in PROBLEMATIC_TEXT:
        if problem in text:
            _handle_exception("text contains {}".format(problem),
                              file_name, client)
            return False
    if collections.Counter(text)['\n'] <= 2:
        _handle_exception("less than 3 text lines", file_name, client)
        return False
    return True


def _get_page_text(client):
    '''@return text of client's HTML page'''
    return client.find_element(marionette_driver.By.TAG_NAME, "body").text


def _kill(*processes):
    '''cleans up after processes'''
    for process in processes:
        if process is not None:
            process.terminate()


def _navigate_or_fail(client, url, file_name, tries=0):
    '''navigates client to url, on failure renames file'''
    try:
        client.navigate(url)
        if _check_text(_get_page_text(client).lower(), file_name, client):
            _write_text(client, file_name)
    except (errors.NoSuchElementException, DelayError, socket.error):
        if tries < 3:
            time.sleep(0.1)
            logging.info("retry %d on %s", tries+1, file_name)
            return _navigate_or_fail(client, url, file_name, tries+1)
        else:
            _handle_exception("failed repeatedly to get page text",
                              file_name, client)
    except errors.TimeoutException as e:
        try:
            if _check_text(_get_page_text(client).lower(), file_name, client):
                _handle_exception(e.message, file_name, client)
        except (errors.NoSuchElementException, CaptureError, DelayError) as e2:
            _handle_exception('after timeout ' + e2.message, file_name, client)
    except (errors.UnknownException, CaptureError) as e:
        _handle_exception(e.message, file_name, client)


def _handle_exception(exception, file_name, client):
    '''renames file to mention exception cause'''
    logging.warn('exception: %s', exception)
    to = '{}_{}'.format(
        file_name,
        str(exception).split('\n')[0].replace(' ', '_').replace(os.sep, '___').replace("'", ''))[:255]
    try:
        os.rename(file_name, to)
    except OSError as e:
        logging.warn("error renaming %s to %s\n%s", file_name, to, e)
    _write_text(client, to)


def _normalize_url(url='google.com'):
    '''normalizes the url by adding the "http://' scheme if none exists'''
    if not urlparse.urlparse(url).scheme:
        url = "http://" + url
    return (url, urlparse.urlparse(url).netloc)


def _open_browser(exe=FIREFOX_PATH + ' -marionette', open_timeout=60):
    '''returns an initialized browser instance with marionette'''
    env_with_debug = os.environ.copy()
    env_with_debug["MOZ_DISABLE_AUTO_SAFE_MODE"] = 'set'
    exewholepath, exeargs = exe.split(' ', 1)
    (exedir, _) = os.path.split(exewholepath)
    # sorry, this is linux/osx-specific, try appending to PATH on windows
    env_with_debug["LD_LIBRARY_PATH"] = '/lib:/usr/lib:' + (
        exedir + '/TorBrowser/Tor')
    _avoid_safe_mode(exedir)
    # print 'lpath: %s' % env_with_debug["LD_LIBRARY_PATH"]
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
        return (subprocess.Popen(['tshark', '-w' + loc, 'host ' + bridge]),
                loc)


def _open_with_timeout(browser, page, timeout=600, burst_wait=3, bridge=None):
    '''navigates browser to url while capturing the packet dump, aborts
    after timeout. If bridge, that is the IP address of the connected
    bridge, just capture traffic to there (need to set this by hand)

    '''
    client = Marionette('localhost', port=2828, socket_timeout=(timeout-30))
    try:
        client.start_session()
    except socket.timeout:
        for process in psutil.process_iter():
            if process.name() == "firefox":
                process.kill()

    (url, domain) = _normalize_url(page)

    (tshark_process, file_name) = _open_packet_dump(domain, bridge)
#    thread = threading.Thread(target=client.navigate, args=(url,))
    thread = threading.Thread(target=_navigate_or_fail,
                              args=(client, url, file_name))
    thread.daemon = True
    thread.start()

    # todo: this is code duplication for both _open functions
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


def _write_text(client, srcfile):
    '''writes web text to file'''
    with open(srcfile + ".text", "a") as f:
        try:
            f.write(_get_page_text(client).encode('utf-8'))
        except:
            f.write('error retrieving text: {}'.format(sys.exc_info()[1]))


class CaptureError(Exception):
    '''raised when capture text is buggy'''
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return repr(self.message)

    def __str__(self):
        return self.message


class DelayError(Exception):
    '''raised when capture needs to wait a bit (text is missing or greeter)'''
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return 'DelayError({!r})'.format(self.message)

    def __str__(self):
        return repr(self)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        browse_to(sys.argv[1])
    else:
        browse_to(sys.argv[1], sys.argv[2])
