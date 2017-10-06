'''configuration variables'''
# logging
import logging

LOGFORMAT = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

#LOGLEVEL = logging.WARN INFO DEBUG
LOGLEVEL = logging.INFO

reload(logging)
logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)
# outlier removal level: 2 is ...
OR_LEVEL = 2
# remove domains with <= MIN_CLASS_SIZE instances
REMOVE_SMALL = True
# combine these
def trace_args():
    '''@return combined values as used in code'''
    return {"remove_small": REMOVE_SMALL, "or_level": OR_LEVEL}
# minimum class size 
MIN_CLASS_SIZE = 30
