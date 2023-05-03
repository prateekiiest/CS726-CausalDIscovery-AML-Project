
from ..functional import graph


def delete(x, y, H, C):
    """
    delete the edge between x and y, and for each h in H:
    (1) delete the previously undirected edge between x and y;
    (2) directing any previously undirected edge between x and h in H as x->h.

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    H: the neighbors of y that are adjacent to x
    C: numpy.array
        CPDAG

    Returns
    -------
    out: numpy.array
        new C
    """

    # first operate
    C[x, y] = 0
    C[y, x] = 0

    # second operate
    C[H, y] = 0

    # third operate
    x_neighbor = graph.neighbors(x, C)
    C[list(H & x_neighbor), y] = 0

    return C


def delete_validity(x, y, H, C):
    """
    check whether a delete operator is valid

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    H: the neighbors of y that are adjacent to x
    C: numpy.array
        CPDAG

    Returns
    -------
    out: bool
        if True denotes the operator is valid, else False.
    """

    na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
    na_yx_h = na_yx - H

    # only one condition
    condition = graph.is_clique(na_yx_h, C)

    return condition
