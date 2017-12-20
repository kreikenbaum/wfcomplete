'''additional metrics to sklearn's'''
from sklearn import metrics, model_selection
import numpy as np

import config

#    p_ = clf.fit(X1, y1).predict_proba(X2)
#    return bounded_auc(y2, p_[:, 1], y_bound, pos_label=0)
## todo: maybe use prettytable
#doesn't really belong here, but neither does it to fit
def binarize(y, keep=-1, transform_to=0):
    '''binarize data in y: transform all values to =transform_to= except =keep=
    >>> list(_binarize([0, -1, 1]))
    [0, -1, 0]
    >>> list(_binarize([0, -1, 1], transform_to=3))
    [3, -1, 3]'''
    for y_class in y:
        if y_class == keep:
            yield keep
        else:
            yield transform_to


def bounded_auc(y_true, y_predict, y_bound, **kwargs):
    '''@return bounded auc of probabilistic prediction, normalized to [0, 1]
    >>> bounded_auc([1, 1, 1, 0], np.array([[1, 1, 1, 0], [0, 0, 0, 1]]).T, 1)
    1.0
    >>> bounded_auc([1, 1, 1, 0], np.array([[1, 1, 1, 0], [0, 0, 0, 1]]).T, .2)
    1.0
    >>> bounded_auc([1, 1, 1, 0], np.array([[0, 0, 0, 1], [1, 1, 1, 0]]).T, 1)
    0.0
    '''
    newfpr, newtpr = bounded_roc(y_true, y_predict, y_bound, **kwargs)
    return metrics.auc(newfpr, newtpr) / y_bound


# todo: examine constants: why does row 0 correspond to label 1?
def bounded_roc(y_true, y_predict, y_bound, **kwargs):
    '''@return (fpr_array, tpr_array) with 0 <= fpr <= y_bound'''
    assert 0 <= y_bound <= 1
    if len(y_predict.shape) == 2 and y_predict.shape[1] == 2:
        y_predict = y_predict[:, 0]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_predict, 1, **kwargs)
    newfpr = [x for x in fpr if x < y_bound]
    newfpr.append(y_bound)
    newtpr = np.interp(newfpr, fpr, tpr)
    return (newfpr, newtpr)


def compute_bounded_auc_score(clf, X, y, y_bound=0.01):
    '''@return cross-validated bounded auc of clf on X and y'''
    scorer = metrics.make_scorer(
        bounded_auc, needs_proba=True, y_bound=y_bound)
    y = list(binarize(y, transform_to=1))
    return 1/y_bound * model_selection.cross_val_score(
        clf, X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM,
        scoring=scorer).mean()
