import logging

LOGFORMAT = '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'

#LOGLEVEL = logging.WARN INFO DEBUG
LOGLEVEL = logging.INFO

reload(logging)
logging.basicConfig(format=LOGFORMAT, level=LOGLEVEL)

def add_file_output(filename):
    sh = logging.FileHandler(filename)
    sh.setFormatter(logging.Formatter(LOGFORMAT))
    logging.getLogger().addHandler(sh)
