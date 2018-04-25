'''additional metrics to sklearn's'''
from sklearn import metrics, model_selection
import numpy as np

import config


# doesn't really belong here, but neither does it to fit
def binarize(y, keep=-1, transform_to=0):
    '''@return list(binarized)'''
    return list(binarized(y, keep, transform_to))


def binarized(y, keep=-1, transform_to=0):
    '''binarized data in y: transform all values to =transform_to= except =keep=
    >>> list(binarized([0, -1, 1]))
    [0, -1, 0]
    >>> list(binarized([0, -1, 1], transform_to=3))
    [3, -1, 3]'''
    for y_class in y:
        if y_class == keep:
            yield keep
        else:
            yield transform_to


def binarize_probability(preds):
    '''binarizes a probability array:
    >>> binarize_probability(np.array([[0.5, 0.2, 0.3], [0.3, 0.2, 0.5]]))
    array([[0.5, 0.5],
           [0.3, 0.7]])'''
    out = np.empty((preds.shape[0], 2))
    out[:, 0] = preds[:, 0]
    out[:, 1] = preds[:, 1:].sum(axis=1)
    return out


def compute_bounded_auc_score(clf, X, y, y_bound=0.01):  # , scorer=None):
    '''@return cross-validated bounded auc of clf on X and y
    @param scorer: used for testing this module's other methods
    @param clf classifier to use
    @param X, y features and labels
    @param y_bound maximum fpr rate for which to consider the auc score
    '''
    # if github sklearn issue #3273 gets pulled
    scorer = metrics.make_scorer(
        metrics.roc_auc_score, greater_is_better=True, needs_threshold=True,
        max_fpr=y_bound)
    return model_selection.cross_val_score(
        clf, X, y, cv=config.FOLDS, n_jobs=config.JOBS_NUM,
        scoring=scorer).mean()


def pos_label(y_true):
    '''@return element in y_true != -1'''
    assert -1 in y_true and len(set(y_true)) == 2
    return [x for x in y_true if x != -1][0]


def tpr_fpr_tpa(confusion_matrix):
    '''@return array of (tpr/recall, fpr, tpa/precision) tuples'''
    TP = np.diag(confusion_matrix) * 1.0
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)
    return zip(TP/(TP+FN), FP/(FP+TN), TP/(TP+FP))
