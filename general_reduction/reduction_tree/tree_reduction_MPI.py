"""
@daniel.mimouni
This code aim is to propose an implementation of the computation of improving an approximate tree, following the method
proposed in 'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler.
"""
import sys
import time
# Multiprocessing
from functools import partial
from multiprocessing import Pool
from mpi4py import MPI
# my codes
from reduction_tree.MAM import *  # my implementation of MAM, that computes the barycenter between probability densities
from reduction_tree.MAM_MPI import *
from reduction_tree.barycenter_IBP import *
from reduction_tree.LP_tree_reduction import *
from reduction_tree.find_process_data import *


def reduction_tree(H, G, Pi=np.zeros((2, 2)), rho=1000, method='LP', lambda_IBP=25, npool=1):
    comm = MPI.COMM_WORLD
    rank = 0 # comm.Get_rank()
    pool_size = 1 #
    pool_sizeMAM = comm.Get_size()
    """
    :param H: Initial tree
    :param G: Approximated tree structure: only the filtration and the quantifiers are necessary
    H and G must have the same number of stages
    :param Pi: Initialize the transport matrix between two tree
    :param method: Method to compute the barycenter problem : LP, MAM, IBP

    :return:
    * G : an approximated tree of H obtained from a rewritting of 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler taking advantage of the barycenter LP problem that enables to
    use the Method of Averaged Marginals (MAM) to solve the Linear Program
    * D_ij : the distance matrix between all nodes of H and G. Note that D_ij[0,0] is the nested distance betwen H and G.
    * Pi : the transport matrix between G and H
    * time_tot: total time in second to compute the reduction
    """

    assert method in ['LP', 'MAM', 'IBP']
    # Time management
    start = time.time()

    # PARAMETERS
    # Careful: this parameter can cause a difference more or less significative between the exact distance_GH(H,G) and the
    # distance computed in this algorithm that prevent conditional probabilities to be null.
    # Note that this difference is also due to the approximation of the iterative Pi_hat from the initialization to the end
    epsilon = 10 ** -3

    ## INITIALIZATION OF PI: this is needed because the recursive computation is based on the previous iteration,
    # therefore at iteration 0 it is based on the initialization
    # I build recursively the transport matrix thanks to the probabilities of the initial tree
    N1 = len(G.nodes)  # nb of nodes in G
    N2 = len(H.nodes)  # nb of nodes in H
    if np.sum(Pi) == 0:
        Pi = np.zeros((N1, N2))
        Pi[0, 0] = 1
        # number of stages
        T = np.max([G.nodes[i]['stage'] for i in G.nodes])
        # Nodes of the stage T-1: so we start the recursivity with them
        ancestor_n = [i for i in G.nodes if G.nodes[i]['stage'] == T - 1]  # list_n = set(ancestor_n) follows
        ancestor_m = [i for i in H.nodes if H.nodes[i]['stage'] == T - 1]
        for t in range(T - 1, -1, -1):
            # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes
            list_m = set(ancestor_m)
            list_n = set(ancestor_n)
            ancestor_m = []
            ancestor_n = []
            for n in list_n:
                for m in list_m:
                    # Children of nodes n and m
                    children_n = [i for i in G.successors(n)]
                    children_m = [i for i in H.successors(m)]

                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    for i in children_m:
                        for j in children_n:
                            Pi[j, i] = H[m][i]['weight'] / len(children_n)

                    # I collect the ancestor of nodes n and m
                    if t > 0:
                        ancestor_n.append([i for i in G.predecessors(n)][0])
                        ancestor_m.append([i for i in H.predecessors(m)][0])

    ## INITAILIZATION OF THE DISTANCE MATRIX : exact process distance between the leaves
    # Distance matrix initialization
    # D_ij = np.zeros((N1, N2))

    ai, wi, pi, T = find_process_data(G)
    aj, wj, pj = find_process_data(H)[:-1]
    # Using Euclidean distance
    leaves_G = len(ai)
    leaves_H = len(aj)
    nodesG = []
    nodesH = []
    D_ij = np.zeros((leaves_G, leaves_H))
    for i in range(leaves_G):
        nodesG.append(ai[i][-1])
        for j in range(leaves_H):
            D_ij[i, j] = np.sum((wi[i] - wj[j]) ** 2)
            if i == 0:
                nodesH.append(aj[j][-1])

    ## OPTIMIZATION OF THE PROBABILITIES
    # Nodes of the stage T-1: so we start the recursivity with them
    ancestor_m = [i for i in H.nodes if H.nodes[i]['stage'] == T - 1]
    ancestor_n = [i for i in G.nodes if G.nodes[i]['stage'] == T - 1]
    record_t_n = []
    nb_n = []
    nb_m = []
    for t in range(T - 1, -1, -1):
        starttime = time.time()
        # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes
        list_m = set(ancestor_m)
        list_n = set(ancestor_n)

        # Parrallel work using MPI:
        outputs = {}
        splitting_work = division_tasks(len(list_n), pool_size)
        for work in splitting_work[rank]:
            n = list(list_n)[work]
            outputs[n] = _loop_subtree_n(t, G, H, nodesG, nodesH, list_m, D_ij, Pi,
                            method, lambda_IBP, rho, epsilon, pool_sizeMAM, n)

        if pool_size > 1:
            l_outputs = comm.gather(outputs, root=0)
            if rank == 0:
                for dico in l_outputs:
                    for key in dico.keys():
                        outputs[key] = dico[key]
            outputs = comm.bcast(outputs, root=0)

        Pi_k = {}
        D_ij_t = []
        ancestor_n = []
        for i_n, n in enumerate(list_n):
            Pi_k[n] = outputs[n]["Pi_k"]
            D_ij_t.append(outputs[n]["D_ij_t"])
            ancestor_n.extend(outputs[n]["ancestor_n"])
            if i_n == 0:
                ancestor_m = outputs[n]["ancestor_m"]

            children_n = [j for j in G.successors(n)]
            for i_m, m in enumerate(list_m):
                children_m = [i for i in H.successors(m)]
                # FILL the transport matrix with the conditional probabilities
                # when the whole tree is treated, I will use (23) from 'Tree approximation for discrete time stochastic process:
                # a process distance approach' from Kovacevic and Pichler to build the Pi(i,j)
                for j_c, j in enumerate(children_n):
                    Pi[j, children_m] = Pi_k[n][i_m][j_c, :]  # transport plan between subtree at node n and subtree at node m(index i_m)

                    # FILL the probabilities in the approximated tree
                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    # I don't need to go through every nodes of a stage of the original tree, only one is sufficient:
                    if i_m == 0:
                        G[n][j]['weight'] = np.sum(Pi[j, children_m])
                        # sum_Gj = sum_Gj + np.abs(G[n][j]['weight'])
                        G[n][j]['weight'] = np.round(G[n][j]['weight'], 3)

        nodesG, nodesH = list(list_n), list(list_m)
        D_ij = np.array(D_ij_t)
        record_t_n.append(np.round(time.time()-starttime, 4)) #/len(list_n))
        nb_m.append(len(list_m))
        nb_n.append(len(list_n))

    # REBUILD the updated Transport matrix between trees H and G
    # Exact method:
    Pi[0, 0] = 1
    list_i1 = [i for i in H.nodes if H.nodes[i]['stage'] == 1]
    list_j1 = [i for i in G.nodes if G.nodes[i]['stage'] == 1]
    for t in range(1, T + 1):
        for i1 in list_i1:
            ancestor_m = [i for i in H.predecessors(i1)]
            for j1 in list_j1:
                ancestor_n = [j for j in G.predecessors(j1)]
                for m in ancestor_m:
                    for n in ancestor_n:
                        Pi[j1, i1] = Pi[j1, i1] * Pi[n, m]
        list_i1 = [i for i in H.nodes if H.nodes[i]['stage'] == t + 1]
        list_j1 = [i for i in G.nodes if G.nodes[i]['stage'] == t + 1]

    # Time management:
    time_tot = time.time() - start
    # Output
    # D_ij[0,0] is the approached Nested distance
    return (G, D_ij[0,0], Pi,record_t_n[::-1], nb_n[::-1], nb_m[::-1], time_tot)


def _loop_subtree_n(t, G, H, nodesG, nodesH, list_m, D_ij, Pi, method, lambda_IBP, rho, epsilon,pool_sizeMAM, n):
    ancestor_m = []
    ancestor_n = []
    D_ij_t = np.zeros(len(list_m))
    b = []  # list of probabilities
    c = {}  # distance matrix with the ponderations
    children_n = [j for j in G.successors(n)]
    children_ni = [j for j in range(len(nodesG)) if nodesG[j] in children_n]
    dist_matrices = []

    # go through each node and treat the subtree (node + its children)
    for i_m, m in enumerate(list_m):
        # Children of nodes m
        children_m = [i for i in H.successors(m)]
        children_mj = [i for i in range(len(nodesH)) if nodesH[i] in children_m]

        # Constraint on p (b is the list of the source probabilities)
        p = np.array([H[m][i]['weight'] for i in children_m])
        b.append(p)

        # Distance matrix
        dist = D_ij[children_ni, :]
        dist = dist[:, children_mj]
        dist_matrices.append(dist)
        # ponderation on the distance matrix due to the initialization/previous iteration of Pi
        c[i_m] = dist * Pi[n, m]

        # I collect the ancestor of node m for next step
        if t > 0:
            ancestor_m.append([i for i in H.predecessors(m)][0])

    # I collect the ancestor of node n for next step
    if t > 0:
        ancestor_n.append([i for i in G.predecessors(n)][0])

    # I treat the resolution of the LP as a barycenter problem, using MAM:
    # this provides the barycenter AND the transport matrices !
    if len(children_n) == 1:
        # this is a trivial case where the subtree of the approximate tree has only one branch (1 chil at the node)
        # then directly:
        Pi_k = [np.expand_dims(bi, axis=0) for bi in b]
    elif len(children_n) > 1:
        if method == 'MAM':
            if pool_sizeMAM > 1:
                resMAM = MAM_MPI(b, M_dist=c, exact=False, rho=rho, keep_track=False, computation_time=10,
                             iterations_min=10, iterations_max=200, precision=10 ** -4, logs=False)
            else:
                resMAM = MAM(b, M_dist=c, exact=False, rho=rho, keep_track=False, computation_time=10,
                             iterations_min=10, iterations_max=200, precision=10 ** -4, logs=False)
            Pi_k = resMAM[1]

        elif method == 'IBP':
            # I reshape the probabilities vector as [p1,p2,p(m),0,0,0,...,0], [0,0,0,p1,p2,p(m),0,0,0,...,0] to keep one distance matrix as built in c_
            sumS = sum(len(sublist) for sublist in b)
            acc = 0
            b_ = []
            c_ = []
            for i_m, p in enumerate(b):
                v = np.zeros(sumS)
                v[acc:acc + len(p)] = p
                b_.append(v)
                acc += len(p)
                c_.append(c[i_m])
            c_ = np.concatenate(c_, axis=1)

            resIBP = barycenter_IBP(b_, c_, computation_time=1, iterations_min=10, iterations_max=200,
                                    lambda_sinkhorn=lambda_IBP, precision=10 ** -4)
            Pi_ibp = resIBP[1]
            acc = 0
            Pi_k = []
            for i_m, p in enumerate(b):
                Pi_k.append(Pi_ibp[i_m][:, acc:acc + len(p)])
                acc += len(p)

        elif method == 'LP':
            c_lp = [c[i_m] for i_m, _ in enumerate(list_m)]
            c_lp = np.concatenate(c_lp, axis=1)
            res_LP = LP_reduction_nt(c_lp, b)
            Pi_lp = res_LP[0]
            acc = 0
            Pi_k = []
            for p in b:
                Pi_k.append(Pi_lp[:, acc:acc + len(p)])
                acc += len(p)

    for i_m, m in enumerate(list_m):
        # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
        Pi_k[i_m] = Pi_k[i_m].clip(epsilon)
        Pi_k[i_m] = Pi_k[i_m] / np.sum(Pi_k[i_m])  # The sum of all Pi[j,i|n,m] for n and m fixed is equal to 1

        # FILL the local distance matrix
        D_ij_t[i_m] = np.sum(np.multiply(Pi_k[i_m], dist_matrices[i_m]))

    # Outputs
    return dict(Pi_k=Pi_k, D_ij_t=D_ij_t, ancestor_m=ancestor_m, ancestor_n=ancestor_n)



def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return (rearranged)





