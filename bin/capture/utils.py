'''common code'''
import os

from sklearn import multiclass, svm

import config


def clf_default(y=None, **svm_params):
    '''@return default classifier with additional params

    set class_weight="balanced" if y represents foreground data'''
    if y is not None and -1 not in y:
        svm_params['class_weight'] = "balanced"
    return multiclass.OneVsRestClassifier(svm.SVC(**svm_params))


def site(status):
    '''@return site-tag to distinguish cloud'''
    try:
        host = status['host']
    except (KeyError, TypeError):
        host = config.OLD_HOST
    if 'duckstein' in host:
        return 'mlsec'
    elif 'pioneering-mode-193216' in host:
        return 'gcloud'
    else:
        raise Exception("unknown host site for %s".format(host))


def path_to(filename):
    '''@return path to filename'''
    return os.path.dirname(os.path.abspath(filename))
