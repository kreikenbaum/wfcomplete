#!/usr/bin/env python
'''create batch/ -dir in current directory (or first argument) second
argument is taken as outlier removal level, if it exists (else
none)
'''
import sys
import counter

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append('.')

    counter.dir_to_cai(sys.argv[1])
