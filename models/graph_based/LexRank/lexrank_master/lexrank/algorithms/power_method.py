import numpy as np
from scipy.sparse.csgraph import connected_components

import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank, pagerank_numpy
# from networkx.convert_matrix import from_numpy_matrix

def _power_method(transition_matrix, increase_power=True):
    '''
    Input:
     transition matrix
     increase_power (bool): If True, each iteration square the transition matrix.

    '''

    # initialize left eigenvectors p of the transition matrix B (i.e Adjacency matrix normalized by row sum
    # so it becomes stohastic) to ones.
    eigenvector = np.ones(len(transition_matrix))

    if len(eigenvector) == 1:
        # the case of a single sentence
        return eigenvector

    transition = transition_matrix.transpose()

    # iterate until convergence
    while True:
        # calculating next p by multiplying it with the transition matrix
        eigenvector_next = np.dot(transition, eigenvector)
        # stop iterating when p is not updated by a lot.
        if np.allclose(eigenvector_next, eigenvector):
            return eigenvector_next
        # update p
        eigenvector = eigenvector_next
        # next time update p, by the square of the, so far, transition matrix
        if increase_power:
            transition = np.dot(transition, transition)


def connected_nodes(matrix):
    '''
    Returns list of connected nodes given an Adjacency Matrix
    '''
    # given a (sparse) binary (NxN) matrix this returns the connected component
    # label of each node in the matrix
    _, labels = connected_components(matrix)

    groups = []
    # for each connected component
    for tag in np.unique(labels):
        # find list of nodes that belong to this connected component
        # print("tag", tag)
        group = np.where(labels == tag)[0]
        # print("grouP", group)
        # and that that list to the list of lists.
        groups.append(group)

    return groups


def stationary_distribution(
    transition_matrix,
    increase_power=True,
    normalized=True,
    damp=False,
    alpha=0.85,
    bias_sim_arr=None
):
    '''
    Calculates the stationary distribution for the markov chain that corresponds
    to the given transition matrix

    Formaly formulated: p^T B = p^T <=> p = B^T where B is the Adjacency matrix
    normalized by row sum so it becomes stohastic

    Eigenvectors will eventualy converge to
    a unique stationary distribution, which corresponds to our sentence sentrality scores,
    so long as the matrix chain that is defined by the B matrix is both aperiodic and irreducible.
    This can be guaranteed when a small probability is added to the transition matrix B,
    in order to ensure that the random walker can escape from periodic and disconnected components.
    Formally, for a uniform random probability of jumping to every node:
    p = [dU + (1-d)B]^T p


    Inputs:
        transition_matrix: This means the adjacency matrix (since dump factor is not implemented?)
        increase_power (bool): If True, after each iteration of the power method, square the transition matrix
        normalized (bool): If true, return normalized sentence centrality scores
    '''
    # number of nodes (sentences)
    size = len(transition_matrix)
    distribution = np.zeros(size)
    if not damp:
        # use power inside each connected component seperately

        # find list of nodes in each connected component
        grouped_indices = connected_nodes(transition_matrix)
        # for each connect component
        for group in grouped_indices:
            # find its transition matrix (without damping):
            t_matrix = transition_matrix[np.ix_(group, group)]
            # find the stationary distribution (eigenvectors) for the nodes inside the c.c
            eigenvector = _power_method(t_matrix, increase_power=increase_power)
            # update stationary distribution for the nodes in the c.c
            distribution[group] = eigenvector

    else:
        # normal power method on the whole graph
        # convert numpy transition matrix to nx graph (unidirected)
        text_graph = nx.convert_matrix.from_numpy_matrix(transition_matrix,
                                                         create_using=nx.DiGraph,
                                                         )
        if bias_sim_arr is not None:
            # pagerank with bias
            # set personalization bias to a format expected by the networkx lib
            bias = dict(zip(range(size), bias_sim_arr))
            distribution = pagerank_numpy(text_graph,
                                          alpha=alpha,
                                          personalization=bias)
            # print("personalization bias is " , bias)
        else:
            # pagerank without bias
            distribution = pagerank_numpy(text_graph,
                                          alpha=alpha)
            # convert to numpy array from dictionary
            distribution = np.array(list(distribution.values()))

    if normalized:
        # normalized stationary distribution scores
        distribution /= size

    # print("distribution is hererere", {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1])})
    return distribution
