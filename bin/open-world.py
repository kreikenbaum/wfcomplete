#! /usr/bin/env python
import logging
import os
import sys

import analyse
import scenario

if __name__ == "__main__":
    logging.basicConfig(format=analyse.LOGFORMAT, level=analyse.LOGLEVEL)
    os.nice(20)
    if len(sys.argv) == 1:
        analyse.open_world(scenario.Scenario(os.getcwd()))
    elif len(sys.argv) > 2:
        logging.warn('only first scenario chosen for open world analysis')
    analyse.open_world(scenario.Scenario(sys.argv[1], smart=True))
