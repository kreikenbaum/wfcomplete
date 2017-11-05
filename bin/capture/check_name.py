#! /usr/bin/env python
'''creates new capture directory and a symlink to it'''
import datetime
import json
import os
import sys

import config

os.chdir(config.SAVETO)

with open('status') as f:
    status = json.load(f)
    enabled = status['addon']['enabled']
    if True in enabled.values():
        print [x for x in enabled if enabled[x]]
        sys.exit(1)
    else:
        newdir = os.path.join('disabled', str(datetime.date.today()))
        os.mkdir(newdir)
        os.symlink(newdir, 'now')
        sys.exit(0)
