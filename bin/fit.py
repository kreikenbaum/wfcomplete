'''helper methods for fitting data, works mostly with X, y sets'''
import collections
import doctest
import logging
from sklearn import cross_validation, grid_search, metrics, multiclass
from sklearn import preprocessing, svm
# ensemble, metrics, neighbors, tree
import numpy as np

import counter

JOBS_NUM = 4
#JOBS_NUM = -3  # 1. maybe -4 for herrmann (2 == -3) used up all memory
FOLDS = 5
#JOBS_NUM = 1; FOLDS = 2 # testing

scaler = None

Result = collections.namedtuple('Result', ['clf', 'best_score_', 'results'])

def _binarize(y, keep=-1, transform_to=0):
    '''binarize data in y: transform all values to =default= except =keep=
    >>> list(_binarize([0, -1, 1]))
    [0, -1, 0]
    >>> list(_binarize([0, -1, 1], transform_to=3))
    [3, -1, 3]'''
    for cls in y:
        if cls == keep:
            yield keep
        else:
            yield transform_to


def _bounded_auc(y_true, y_predict, bound=0.01, **kwargs):
    '''@return bounded auc of (probabilistic) fitted classifier on data.'''
    newfpr, newtpr = _bounded_roc(y_true, y_predict, bound, **kwargs)
    return metrics.auc(newfpr, newtpr)


def _bounded_roc(y_true, y_predict, bound=0.01, **kwargs):
    '''@return (fpr, tpr) within fpr-bounds'''
    assert 0 <= bound <= 1
    if y_predict.shape[1] == 2:
        y_predict = y_predict[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_predict, **kwargs)
    # plot_data.roc(fpr, tpr).savefig('/tmp/roc.pdf')
    # plt.close()
    newfpr = [x for x in fpr if x < bound]
    newfpr.append(bound)
    newtpr = np.interp(newfpr, fpr, tpr)
    return (newfpr, newtpr)


def _bounded_auc_eval(X, y, clf, y_bound=0.1):
    '''evaluate clf X, y, give bounded auc score, 0 is positive class label'''
    X = _scale(X, clf)
    y = list(_binarize(y, transform_to=1))
    #     X, y, train_size=1. / 2, stratify=y)
    scorer = metrics.make_scorer(_bounded_auc, needs_proba=True, bound=y_bound)
    return cross_validation.cross_val_score(
        clf, X, y, cv=FOLDS, n_jobs=JOBS_NUM, scoring=scorer).mean()
#    p_ = clf.fit(X1, y1).predict_proba(X2)
#    return _bounded_auc(y2, p_[:, 1], y_bound, pos_label=0)


def _eval(X, y, clf, folds=FOLDS):
    '''evaluate estimator on X, y, @return result (ndarray)'''
    X = _scale(X, clf)
    return cross_validation.cross_val_score(clf, X, y, cv=folds,
                                            n_jobs=JOBS_NUM).mean()


def _lb(*args, **kwargs):
    '''facade for _binarize, list wrap'''
    return list(_binarize(*args, **kwargs))


def _scale(X, clf):
    '''ASSUMPTION: svc never called on two separate data sets in
    sequence.  That is: _scale(X1_train, svc), _scale(X1_test, svc),
    _scale(Y2_train, svc), _scale(Y2_test, svc), will not
    happen. (without a _scale(..., non_svc) in between). The first is
    treated as training data, the next/others as test.

    @return scaled X if estimator is SVM, else just X
    '''
    global scaler
    if 'SVC' in str(clf):
        logging.debug("_scaled on %s", clf)
        if not scaler:
            scaler = preprocessing.MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            logging.debug("reused scaler")
            return scaler.transform(X)
    else:
        scaler = None
        return X


def _sci_best_at_border(grid_clf):
    '''@return True if best params are at parameter grid borders'''
    c_borders = (grid_clf.param_grid['estimator__C'][0],
                 grid_clf.param_grid['estimator__C'][-1])
    g_borders = (grid_clf.param_grid['estimator__gamma'][0],
                 grid_clf.param_grid['estimator__gamma'][-1])
    return (grid_clf.best_params_['estimator__C'] in c_borders
            or grid_clf.best_params_['estimator__gamma'] in g_borders)


def _sci_fit(C, gamma, step, X, y, scoring=None, probability=False):
    '''@return appropriate gridsearchcv, fitted with X and y'''
    logging.info('C: %s, gamma: %s, step: %s', C, gamma, step)
    cs = _search_range(C, step)
    gammas = _search_range(gamma, step)
    clf = grid_search.GridSearchCV(
        estimator=multiclass.OneVsRestClassifier(
            svm.SVC(class_weight="balanced", probability=probability)),
        param_grid={"estimator__C": cs, "estimator__gamma": gammas},
        n_jobs=JOBS_NUM, verbose=0, cv=FOLDS, scoring=scoring)
    return clf.fit(X, y)


def _search_range(best_param, step=1):
    '''@return new array of parameters to search in, with logspaced steps'''
    _step = 2.**step
    return [best_param / (_step**2), best_param / _step, best_param,
            best_param * _step, best_param * _step**2]


def _stop(y, step, result, previous, C=1, best=1):
    '''@return True if grid should stop

    >>> _stop([1,1,2,2,3,3], 0.0001, 0.5, 0.4) # stop due to step
    True
    >>> _stop([1,2], 1, 0.5, []) # no stop
    False
    >>> _stop([1,2], 1, 0.5, [1,2,3]) # no stop
    False
    >>> _stop([1,2], 1, 1, [1,1,1,1]) # stop due to results
    True
    >>> _stop([1,2], 1, 0.5, [], 1e-200) # stop due to C
    True
    '''
    return (step < 0.001 or
            C < 1e-50 or
            (len(previous) > 3 and  # some tries and with same val and > guess
             max([abs(x - result) for x in previous[-3:]]) < 0.00001 and
             result > best * 1.1 * max(collections.Counter(y).values()) / len(y)))


def helper(counter_dict, outlier_removal=True, cumul=True, folds=FOLDS):
    '''@return grid-search on counter_dict result (clf, results)'''
    if outlier_removal:
        counter_dict = counter.outlier_removal(counter_dict)
    if cumul:
        (X, y, _) = counter.to_features_cumul(counter_dict)
        return my_grid(X, y, folds=folds)
    else:  # panchenko 1
        (X, y, _) = counter.to_features(counter_dict)
        return my_grid(X, y, C=2**17, gamma=2**-19, folds=folds)


def my_grid(X, y, C=2**14, gamma=2**-10, step=4, results=None,
            auc_bound=None, previous=None, folds=FOLDS):
    '''@param results are previously computed results {(C,gamma): accuracy, ...}
    @param auc_bound if this is set, use the bounded auc score with this y_bound
    @return Result(clf, best_score_, results) (namedtuple see above)'''
    if not results:
        previous = []
        results = {}
        scaler = None # guesstimate: one grid search per data set (TODO: refact)
    bestclf = None
    bestres = np.array([0])
    for c in _search_range(C, step):
        for g in _search_range(gamma, step):
            clf = multiclass.OneVsRestClassifier(svm.SVC(
                gamma=g, C=c, class_weight='balanced',
                probability=(True if auc_bound else False)))
            if (c, g) in results:
                current = results[(c, g)]
            else:
                if auc_bound:
                    current = _bounded_auc_eval(X, y, clf, auc_bound)
                else:
                    current = _eval(X, y, clf, folds=folds)
                results[(c, g)] = current
            if not bestclf or bestres < current:
                bestclf = clf
                bestres = current
            logging.info('c: %8s g: %15s res: %.6f', c, g, current.mean())
    previous.append(bestres.mean())
    if _stop(y, step, bestres.mean(), previous, C, best=(auc_bound if
                                                         auc_bound else 1)):
        logging.info('grid result: %s', bestclf)
        return Result(bestclf, bestres.mean(), results)
    if (bestclf.estimator.C in (_search_range(C, step)[0],
                                _search_range(C, step)[-1])
            or bestclf.estimator.gamma in (_search_range(gamma, step)[0],
                                           _search_range(gamma, step)[-1])):
        logging.warn('optimal at border. c:%f, g:%f, score: %f',
                     bestclf.estimator.C, bestclf.estimator.gamma, bestres)
    else:
        step /= 2.
    return my_grid(X, y, bestclf.estimator.C, bestclf.estimator.gamma,
                   step, results, previous=previous,
                   auc_bound=auc_bound)


def roc(clf, X_train, y_train, X_test, y_test):
    '''@return (fpr, tpr, thresholds) of =clf= on adjusted data

    It uses the same scaling and binarization as my_grid.
    '''
    fitted = clf.fit(_scale(X_train, clf), _lb(y_train, transform_to=1))
    prob = fitted.predict_proba(_scale(X_test, clf))
    fpr, tpr, thresh = metrics.roc_curve(
        _lb(y_test, transform_to=1), prob[:, 1], 1)
    return fpr, tpr, thresh, prob


def sci_grid(X, y, C=2**14, gamma=2**-10, step=2, scoring=None,
             probability=False):
    '''(scikit-)grid-search on fixed params, searching laterally and in depth

    @param X,y,C,gamma,folds as for the classifier
    @param step exponential step size, c-range = [c/2**step, c, c*2**step], etc
    @param grid_args: arguments for grid-search, f.ex. scorer

    @return gridsearchcv classifier (with .best_score and .best_params)
    >>> test = sci_grid([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], [0, 0, 0, 1, 1, 1], 0.0001, 0.000001); test.best_score_
    1.0
    '''
    previous = []

    clf = _sci_fit(C, gamma, step, X, y, scoring, probability)
    while not _stop(y, step, clf.best_score_, previous):
        if _sci_best_at_border(clf):
            pass  # keep step, search laterally
        else:
            step = step / 2.
        previous.append(clf.best_score_)
        clf = _sci_fit(clf.best_params_['estimator__C'],
                       clf.best_params_['estimator__gamma'],
                       step, X, y)
    return clf


doctest.testmod()

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    print 'nothing to do, avoids joblib warning'
