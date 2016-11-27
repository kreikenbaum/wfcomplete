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
    try:
        if len(sys.argv) == 3:
            counter.dir_to_wang(sys.argv[1],
                                outlier_removal_lvl=int(sys.argv[2]))
        else:
            counter.dir_to_wang(sys.argv[1])
    except OSError:
        print("dir already exists")
        system.exit()
