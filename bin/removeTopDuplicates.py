#! /usr/bin/env python
'''removes duplicates from top-1m sites
- collect all sites with same non-top-level-domain (yandex.ru/yandex.com/...)
- remove second to last instance
'''
import collections


def remove_dupes(filename):
    with open(filename) as f:
        dupes = collect_bases(f)
    return remove_bases(filename, dupes)


def collect_bases(readable):
    '''@return all duplicates of site (like google.co, google.co.uk, ...)'''
    dupes = collections.defaultdict(lambda: [])
    for line in readable:
        idx, site = line.split(',')
        parts = site.split('.')
        if parts[0] == 'www':
            if len(parts) > 2:
                parts.pop(0)
        dupes[parts[0]].append(idx)
    return dupes


def remove_bases(filename, dupes):
    '''@yield line from filename if not dupe'''
    with open(filename) as readable:
        dupe_ids = set()
        for its_indexes in dupes.values():
            dupe_ids.update(its_indexes[1:])
        for line in readable:
            idx, site = line.split(',')
            if idx in dupe_ids:
                continue
            yield line
