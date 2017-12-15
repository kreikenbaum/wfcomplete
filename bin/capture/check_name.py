#! /usr/bin/env python
'''checks setting, creates new capture directory and a /now/-symlink to it'''
import datetime
import logging
import json
import os
import sys

import config

#CHECKS={ ## later: map name to (lambda?): check these aspects

def mkdiretc(dirname, prefix=''):
    '''creates directory "name"/date, and a "now"-symlink'''
    try:
        os.mkdir(dirname)
    except OSError: pass
    try:
        newdir = os.path.join(dirname, prefix + str(datetime.date.today()))
        os.mkdir(newdir)
        os.symlink(newdir, 'now')
        sys.exit(0)
    except OSError:
        logging.warn("%s already exists", newdir or "directory")
        sys.exit(0)

if __name__ == "__main__":
    os.chdir(config.SAVETO)

    with open('status') as f:
        status = json.load(f)
        prefix = ''
        if status['config']['bridge'] == " (bridge unknown)":
            prefix = 'no'
        prefix += 'bridge'
        enabled = status['addon']['enabled']
        for (addon, is_enabled) in enabled.iteritems():
            if is_enabled:
                if addon == '@wf-cover':
                    if status['addon']['factor']:
                        prefix += '-{}aI'.format(status['addon']['factor'])
                    else:
                        prefix += '-50aI'
                name = addon.replace("@", '')
                break # better safe than sorry
        else:
            name = "disabled"
        mkdiretc(name, prefix + '--')
