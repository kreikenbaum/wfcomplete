#! /usr/bin/env python
'''prints whether the wf-cover addon is enabled or not'''
import json
import os

prefline = os.popen('get_pref.sh "extensions.xpiState"').read()
state = json.loads(json.loads(prefline))
if state['app-profile']['@wf-cover']['e']:
    print 'addon enabled'
else:
    print 'addon disabled'
