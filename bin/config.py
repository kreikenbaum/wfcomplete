'''configuration variables'''
import logconfig

from capture import config as capture_config

# ### SCENARIO
# ## SCENARIO NAME WITH DEFENSE
COVER_NAME = capture_config.COVER_NAME

OLD_HOST = capture_config.OLD_HOST

# ## TRACE EXTRACTION
MAIN = capture_config.MAIN

# ### OUTLIER REMOVAL
# minimum class size
MIN_CLASS_SIZE = 30
# level of outlier removal
OR_LEVEL = 2
# remove domains with <= MIN_CLASS_SIZE instances
REMOVE_SMALL = True
# combine these
def trace_args():
    '''@return combined values as used in code'''
    return {"remove_small": REMOVE_SMALL, "or_level": OR_LEVEL}


# #### CROSS-VALIDATION
FOLDS = 10
JOBS_NUM = -3
VERBOSE = 0
# JOBS_NUM = -3  # 1. maybe -4 for herrmann (2 == -3) used up all memory
# ### TESTING / DEBUGGING
# JOBS_NUM = 1; FOLDS = 2
# ## copy&paste:
# config.JOBS_NUM = 1; config.FOLDS = 2; config.VERBOSE = 1
# config.JOBS_NUM = 3; config.FOLDS = 3; config.VERBOSE = 3
