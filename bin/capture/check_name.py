#! /usr/bin/env python
'''checks setting, creates new capture directory and a /now/-symlink to it'''
import datetime
import logging
import json
import os
import sys

import config


def mkdiretc(type_name, prefix='', suffix=''):
    '''creates directory "type_name"/prefix--date--suffix, plus "now"-symlink'''
    try:
        os.mkdir(type_name)
    except OSError: pass
    try:
        if prefix:
            scenario_name = '{}--{}'.format(prefix, datetime.date.today())
        else:
            scenario_name = str(datetime.date.today())
        if suffix:
            scenario_name += '--' + suffix
        newdir = os.path.join(type_name, scenario_name)
        os.mkdir(newdir)
    except OSError:
        logging.warn("%s already exists", newdir or "directory")
        if newdir: print newdir
        sys.exit(0)
    try:
        os.symlink(newdir, 'now')
    except OSError:
        os.remove('now')
        os.symlink(newdir, 'now')
    print newdir
    sys.exit(0)


def check(status, name):
    '''raise exception if status does not match name, warn if maybe off'''
    enabled_defenses = [addon for (addon, enabled) in status['addon']['enabled'].iteritems() if enabled]
    ## (a) == (b) in python is "a iff b" (stackoverflow.com/questions/34157836)
    if (name == config.COVER_NAME) != status['local-servers']['cover-traffic']:
        logging.warn("localhost cover server %s does not match scenario %s",
                     status['local-servers']['cover-traffic'], name)
    assert ((name == "wtf-pad") == status['local-servers']['wtf-pad'] == status['bridge-servers']['wtf-pad'])
    assert (name == "disabled") == (len(enabled_defenses) == 0)


if __name__ == "__main__":
    os.chdir(config.SAVETO)

    if len(sys.argv) == 3:
        suffix = sys.argv[1] + '@' + sys.argv[2]
    else:
        suffix = ''

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
        check(status, name)
        mkdiretc(name, prefix, suffix)
