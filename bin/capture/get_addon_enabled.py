#! /usr/bin/env python
'''prints whether the wf-cover addon is enabled or not'''
import json
import os

prefline = os.popen('get_pref.sh "extensions.xpiState"').read()
state = json.loads(json.loads(prefline))
if state['app-profile']['@wf-cover']['e']:
    print 'enabled'
    exit(0)
else:
    print 'disabled'
    exit(1)
