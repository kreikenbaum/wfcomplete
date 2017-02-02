'''helper methods for fitting data'''
import collections
import doctest
import logging
from sklearn import cross_validation, grid_search, metrics, multiclass, preprocessing, svm
import numpy as np
# ensemble, metrics, neighbors, tree

import counter

#JOBS_NUM = -3  # 1. maybe -4 for herrmann (2 == -3) used up all memory
JOBS_NUM = -3

scaler = None

Result = collections.namedtuple('Result', ['clf', 'best_score_',
                                           'results', 'name'])

def _binarize(y, keep=-1, default=0):
    '''binarize data in y: transform all values to =default= except =keep=
    >>> list(_binarize([0, -1, 1]))
    [0, -1, 0]'''
    for el in y:
        if el == keep:
            yield keep
        else:
            yield default


def _bounded_auc(y_true, y_predict, bound=0.01, **kwargs):
    '''@return bounded auc of (probabilistic) fitted classifier on data.'''
    newfpr, newtpr = _bounded_roc(y_true, y_predict, bound, **kwargs)
    return metrics.auc(newfpr, newtpr)


def _bounded_roc(y_true, y_predict, bound=0.01, **kwargs):
    '''@return (fpr, tpr) within fpr-bounds'''
    assert 0 <= bound <= 1
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_predict, **kwargs)
    newfpr = [x for x in fpr if x < bound]
    newfpr.append(bound)
    newtpr = np.interp(newfpr, fpr, tpr)
    return (newfpr, newtpr)


def _bounded_auc_eval(X, y, clf, y_bound=0.1):
    '''evaluate clf X, y, give bounded auc score, 0 is positive class label'''
    X = _scale(X, clf)
    y = list(_binarize(y))
    (X1, X2, _, y1, y2, _) = _tvts(X, y)
    clf.fit(X1, y1)
    p_ = clf.predict_proba(X2)
    return _bounded_auc(y2, p_[:, 1], y_bound, pos_label=0)


def _clf_name(clf):
    '''@return name of estimator class'''
    return str(clf.__class__).split('.')[-1].split("'")[0]


def _eval(X, y, clf, nj=JOBS_NUM, folds=5):
    '''evaluate estimator on X, y, @return result (ndarray)'''
    X = _scale(X, clf)
    return cross_validation.cross_val_score(clf, X, y, cv=folds, n_jobs=nj).mean()


def _fit(C, gamma, step, X, y, cv, scoring=None, probability=False):
    '''@return appropriate gridsearchcv, fitted with X and y'''
    logging.info('C: %s, gamma: %s, step: %s', C, gamma, step)
    cs = _new_search_range(C, step)
    gammas = _new_search_range(gamma, step)
    clf = grid_search.GridSearchCV(
        estimator=multiclass.OneVsRestClassifier(
            svm.SVC(class_weight="balanced", probability=probability)),
        param_grid={"estimator__C": cs, "estimator__gamma": gammas},
        n_jobs=JOBS_NUM, verbose=0, cv=cv, scoring=scoring)
    return clf.fit(X, y)


def _new_search_range(best_param, step=1):
    '''@return new array of parameters to search in, with logspaced steps'''
    _step = 2.**step
    return [best_param / (_step**2), best_param / _step, best_param,
            best_param * _step, best_param * _step**2]


def _scale(X, clf):
    '''ASSUMPTION: svc never called on two separate data sets in
    sequence.  That is: _scale(X_train, svc), _scale(X_test, svc),
    _scale(Y_train, svc), _scale(Y_test, svc), will not
    happen. (without a _scale(..., non_svc) in between). The first is
    treated as training data, the next/others as test.

    @return scaled X if estimator is SVM, else just X
    '''
    global scaler
    if 'SVC' in str(clf):
        logging.debug("_scaled on %s", _clf_name(clf))
        if not scaler:
            scaler = preprocessing.MinMaxScaler()
            return scaler.fit_transform(X)
        else:
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


def _stop(y, step, result, previous):
    '''@return True if grid should stop

    >>> _stop([1,1,2,2,3,3], 0.0001, 0.5, 0.4) # stop due to step
    True
    >>> _stop([1,2], 1, 0.5, []) # no stop
    False
    >>> _stop([1,2], 1, 0.5, [1,2,3]) # no stop
    False
    >>> _stop([1,2], 1, 1, [1,1,1,1]) # stop due to results
    True
    '''
    return (step < 0.001 or
            (len(previous) > 3 and  # some tries and with same val and > guess
             max([abs(x - result) for x in previous[-3:]]) < 0.001 and
             result > 1.1 * max(collections.Counter(y).values()) / len(y)))


def _tvts(X, y):
    '''@return X1, X2, X3, y1, y2, y3 with each 1/3 of the data (train,
validate, test)
    >> _tvts([[1], [1], [1], [2], [2], [2]], [1, 1, 1, 2, 2, 2])
    ([[1], [2]], [[1], [2]], [[1], [2]], [1, 2], [1, 2], [1, 2]) # modulo order
    '''
    X1, Xtmp, y1, ytmp = cross_validation.train_test_split(
        X, y, train_size=1. / 3, stratify=y)
    X2, X3, y2, y3 = cross_validation.train_test_split(
        Xtmp, ytmp, train_size=.5, stratify=ytmp)
    return (X1, X2, X3, y1, y2, y3)


def helper(counter_dict, outlier_removal=True, n_jobs=JOBS_NUM,
           cumul=True, folds=5):
    '''@return grid-search on counter_dict result (clf, results)'''
    if outlier_removal:
        counter_dict = counter.outlier_removal(counter_dict)
    if cumul:
        (X, y, _) = counter.to_features_cumul(counter_dict)
        return my_grid(X, y, folds=folds, n_jobs=n_jobs)
    else:  # panchenko 1
        (X, y, _) = counter.to_features(counter_dict)
        return my_grid(X, y, C=2**17, gamma=2**-19, n_jobs=n_jobs)


def my_grid(X, y, C=2**14, gamma=2**-10, step=2, results={}, auc_bound=None,
            n_jobs=JOBS_NUM, folds=5, previous=[]):
    '''@param results are previously computed results {(C,gamma): accuracy, ...}
    @param auc_bound if this is set, use the bounded auc score with this y_bound
       @return tuple (optimal_classifier, results_object)'''
    best = None
    bestres = np.array([0])
    cs = _new_search_range(C, step)
    gammas = _new_search_range(gamma, step)
    probability = True if auc_bound else False
    for c in cs:
        for g in gammas:
            clf = multiclass.OneVsRestClassifier(svm.SVC(
                gamma=g, C=c, class_weight='balanced', probability=probability),
                                                 n_jobs=n_jobs)
            if (c, g) in results:
                current = results[(c, g)]
            else:
                if auc_bound:
                    current = _bounded_auc_eval(X, y, clf, auc_bound)
                else:
                    current = _eval(X, y, clf, folds=folds)
                results[(c, g)] = current
            if not best or bestres < current:
                best = clf
                bestres = current
#            logging.debug('c: {:8} g: {:10} acc: {}'.format(c, g,
            logging.info('c: %8s g: %10s acc: %f', c, g, current.mean())
    previous.append(bestres.mean())
    if _stop(y, step, bestres.mean(), previous):
        logging.info('grid result: %s', best)
        return Result(best, bestres.mean(), results, _clf_name(best))
    if (best.estimator.C in (cs[0], cs[-1])
            or best.estimator.gamma in (gammas[0], gammas[-1])):
        logging.warn('optimal parameters found at the border. c:%f, g:%f',
                     best.estimator.C, best.estimator.gamma)
    else:
        step /= 2.
    return my_grid(X, y, best.estimator.C, best.estimator.gamma, step,
                   results, previous=previous, auc_bound=auc_bound,
                   n_jobs=n_jobs, folds=folds)


def sci_grid(X, y, C=2**14, gamma=2**-10, folds=3, step=2):
    '''(scikit-)grid-search on fixed params, searching laterally and in depth

    @param X,y,C,gamma,folds as for the classifier
    @param step exponential step size, c-range = [c/2**step, c, c*2**step], etc
    @param grid_args: arguments for grid-search, f.ex. scorer

    @return gridsearchcv classifier (with .best_score and .best_params)
    >>> test = sci([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], [0, 0, 0, 1, 1, 1], 0.0001, 0.000001, folds=2); test.best_score_
    1.0
    '''
    previous = []

    clf = _fit(C, gamma, step, X, y, folds)
    while not _stop(y, step, clf.best_score_, previous):
        if _sci_best_at_border(clf):
            pass  # keep step, search laterally
        else:
            step = step / 2.
        previous.append(clf.best_score_)
        clf = _fit(clf.best_params_['estimator__C'],
                   clf.best_params_['estimator__gamma'],
                   step, X, y, folds)
    return clf


doctest.testmod()
