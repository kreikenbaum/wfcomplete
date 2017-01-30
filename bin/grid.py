import collections
import doctest
import logging
from sklearn import grid_search, multiclass, svm  # cross_validation, ensemble, metrics, neighbors, preprocessing, tree
import counter

JOBS_NUM = -3  # 1. maybe -4 for herrmann (2 == -3) used up all memory


def _best_at_border(grid_clf):
    '''@return True if best params are at parameter grid borders'''
    c_borders = (grid_clf.param_grid['estimator__C'][0],
                 grid_clf.param_grid['estimator__C'][-1])
    g_borders = (grid_clf.param_grid['estimator__gamma'][0],
                 grid_clf.param_grid['estimator__gamma'][-1])
    return (grid_clf.best_params_['estimator__C'] in c_borders
            or grid_clf.best_params_['estimator__gamma'] in g_borders)


def _fit(c, gamma, step, X, y, cv, scoring=None, probability=False):
    '''@return appropriate gridsearchcv, fitted with X and y'''
    logging.info('c: %s, gamma: %s, step: %s', c, gamma, step)
    cs = _new_search_range(c, step)
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


def helper(counter_dict, outlier_removal=True, nj=JOBS_NUM,
            cumul=True, folds=5):
    '''@return grid-search on counter_dict result (clf, results)'''
    if outlier_removal:
        counter_dict = counter.outlier_removal(counter_dict)
    if cumul:
        (X, y, _) = counter.to_features_cumul(counter_dict)
        return _my(X, y, folds=folds)
    else:  # panchenko 1
        (X, y, _) = counter.to_features(counter_dict)
        return _my(X, y,
                   cs=np.logspace(15, 19, 3, base=2),
                   gammas=np.logspace(-17, -21, 3, base=2))


def my(X, y, cs=_new_search_range(2**14, 2),
        gammas=_new_search_range(2**-10, 2), results={},
        num_jobs=JOBS_NUM, folds=5, probability=False):
    '''@param results are previously computed results {(c, g): accuracy, ...}
       @return tuple (optimal_classifier, results_object)'''
    for c in cs:
        for g in gammas:
            clf = multiclass.OneVsRestClassifier(svm.SVC(
                gamma=g, C=c, class_weight='balanced', probability=probability))
            if (c,g) in results:
                current = results[(c,g)]
            else:
                current = _test(X, y, clf, num_jobs, folds=folds)
                results[(c, g)] = current
            if not best or bestres.mean() < current.mean():
                best = clf
                bestres = current
            logging.debug('c: {:8} g: {:10} acc: {}'.format(c, g,
                                                            current.mean()))
    if (best.estimator.C in (cs[0], cs[-1])
        or best.estimator.gamma in (gammas[0], gammas[-1])):
        logging.warn('optimal parameters found at the border. c:%f, g:%f',
                     best.estimator.C, best.estimator.gamma)
        return _my_grid(X, y,
                       _new_search_range(best.estimator.C),
                       _new_search_range(best.estimator.gamma),
                       results)
    else:
        logging.info('grid result: {}'.format(best))
        return best, results


def sci(X, y, c=2**14, gamma=2**-10, folds=3, step=2):
    '''(scikit-)grid-search on fixed params, searching laterally and in depth

    @param X,y,c,gamma,folds as for the classifier
    @param step exponential step size, c-range = [c/2**step, c, c*2**step], etc
    @param grid_args: arguments for grid-search, f.ex. scorer

    @return gridsearchcv classifier (with .best_score and .best_params)
    >>> test = _my([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], [0, 0, 0, 1, 1, 1], 0.0001, 0.000001); test.best_score_
    1.0
    '''
    previous = []

    clf = _fit(c, gamma, step, X, y, folds)
    while not _stop(y, step, clf.best_score_, previous):
        if _best_at_border(clf):
            pass  # keep step, search laterally
        else:
            step = step / 2.
        previous.append(clf.best_score_)
        clf = _fit(clf.best_params_['estimator__C'],
                   clf.best_params_['estimator__gamma'],
                   step, X, y, folds)
    return clf


doctest.testmod()
