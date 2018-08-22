'''configuration variables'''
import os
import sys

import logconfig

from capture import config as capture_config
from capture import utils

# ### CROSS-VALIDATION
FOLDS = 10
JOBS_NUM = -2  # maybe -4 for herrmann (2 == -3) used up all memory
VERBOSE = 0

# ### CAPTURE
DURATION_LIMIT = capture_config.DURATION_LIMIT


# ### OUTLIER REMOVAL
# minimum class size
MIN_CLASS_SIZE = 30
# level of outlier removal
OR_LEVEL = 2
# remove domains with <= MIN_CLASS_SIZE instances
REMOVE_SMALL = True
# remove domains with duration close to DURATION_LIMIT
REMOVE_TIMEOUT = False
# combine these
def trace_args():
    '''@return combined values as used in code'''
    return ta_helper(REMOVE_SMALL, OR_LEVEL, REMOVE_TIMEOUT)
def ta_helper(rs, orl, rt):
    return {"remove_small": rs, "or_level": orl, "remove_timeout": rt}

# ### SCENARIO
# ## SCENARIO NAME WITH DEFENSE
COVER_NAME = capture_config.COVER_NAME

OLD_HOST = capture_config.OLD_HOST

# ### SITES
SITES = capture_config.SITES.replace("$HOME", os.getenv("HOME"))
try:
    open(SITES)
except IOError:
    SITES = os.path.join(utils.path_to(__file__), '..',
                         capture_config.SITES.replace("$HOME/", ""))

# ### TRACE EXTRACTION
MAIN = capture_config.MAIN


# ### TESTING / DEBUGGING
# JOBS_NUM = 1; FOLDS = 2
# ## copy&paste:
# config.JOBS_NUM = 1; config.FOLDS = 2; config.VERBOSE = 1
# import config; config.JOBS_NUM = 3; config.FOLDS = 3; config.VERBOSE = 3
