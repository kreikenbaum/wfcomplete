#! /usr/bin/env python
'''prints whether the wf-cover addon is enabled or not'''
import json
import os

prefline = os.popen('get_pref.sh "extensions.xpiState"').read()
state = json.loads(json.loads(prefline))
out = {}
for name in state['app-profile']:
    if name in ['tor-launcher@torproject.org',
                'https-everywhere-eff@eff.org',
                'torbutton@torproject.org',
                '{73a6fe31-595d-460b-a920-fcc0f8843232}']:
        continue
    if state['app-profile'][name]['e']:
        out[name] = True
    else:
        out[name] = False
print out
