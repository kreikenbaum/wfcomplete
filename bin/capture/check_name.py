#! /usr/bin/env python
'''checks setting, creates new capture directory and a /now/-symlink to it'''
import datetime
import json
import os
import sys

import config

#CHECKS={


def mkdiretc(name):
    newdir = os.path.join(name, str(datetime.date.today()))
    os.mkdir(newdir)
    os.symlink(newdir, 'now')
    sys.exit(0)

if __name__ == "__main__":
    os.chdir(config.SAVETO)

    with open('status') as f:
        status = json.load(f)
        enabled = status['addon']['enabled']
        for (addon, is_enabled) in enabled.iteritems():
            if is_enabled:
                mkdiretc(addon.replace("@", ''))
        else:
            mkdiretc("disabled")
