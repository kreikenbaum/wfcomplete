'''generalized (=with weights) Damerau-Levenshtein distance'''
import doctest 
import numpy as np

# see wiki
def metric(x, y, weight_insert=1, weight_delete=1, weight_alter=1, weight_transpose=1):
    """
    generalized damerau-levenshtein distance
    > gdl('peacemakers', 'cheesemakers')
    4.0

    the parameters: weight_* set the cost of operations:
    > gdl([10, 35, 70], [70, 35], weight_transpose = 0.1)
    1.1000000000000001
    """
    # why? 1: allows strings, etc, 2: in original, xt, yt are 1-indexed
    xt = (0,) + tuple(x)
    yt = (0,) + tuple(y)

    D = np.empty((len(y) +1, len(x) +1), np.dtype('float64'))
    # D[0,0] is assigned twice 
    D[0] = range(len(x)+1)
    D[:,0] = range(len(y)+1)

    # optimization 1: move the non-transpose step to a seperate loop
    for i in range(1, len(x) +1):
        for j in range(1, len(y) +1):
            costs = []
            if xt[i] == yt[j]:
                costs.append(D[j-1, i-1])
            if (i, j) > (1,1) and xt[i-1] == yt[j] and yt[j-1] == xt[i]:
                costs.append(D[j-2, i-2] + weight_transpose)
            costs.append(D[j-1, i-1] + weight_alter)
            costs.append(D[j-1, i] + weight_insert)
            costs.append(D[j, i-1] + weight_delete)
            D[j, i] = min(costs)

    return D[-1,-1]

if __name__ == "__main__":
    doctest.testmod()
