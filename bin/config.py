'''configuration variables'''
### logging
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

### CROSS-VALIDATION
FOLDS = 10
