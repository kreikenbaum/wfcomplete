#! /usr/bin/env python
'''prints whether the wf-cover addon is enabled or not'''
import json
import os

prefline = os.popen('get_pref.sh "extensions.xpiState"').read()
try:
    state = json.loads(json.loads(prefline))
except TypeError:
    state = {"app-profile": []}
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
print json.dumps(out)
