#!/usr/bin/env python
'''create batch/ -dir in current directory (or first argument)'''
import sys
import counter

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append('.')

    counter.dir_to_panchenko(sys.argv[1])
