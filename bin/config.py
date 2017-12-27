'''configuration variables'''
#### LOGGING
import logging

LOGFORMAT = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

#LOGLEVEL = logging.WARN INFO DEBUG
LOGLEVEL = logging.INFO

reload(logging)
logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

#### OUTLIER REMOVAL
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

#### CROSS-VALIDATION
FOLDS = 10
JOBS_NUM = -3
# JOBS_NUM = -3  # 1. maybe -4 for herrmann (2 == -3) used up all memory
### TESTING
#JOBS_NUM = 1; FOLDS = 2
## copy&paste:
#config.JOBS_NUM = 1; config.FOLDS = 2
