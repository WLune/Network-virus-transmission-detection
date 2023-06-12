import math
from decimal import Decimal

import networkx as nx
import numpy as np


class Method:
    subgraph = nx.Graph()
    source = ''
    method_name = ''
    data = ''

    def __init__(self):
        self.method_name = self.__class__
        self.reset_centrality()

    def __init__(self):
        self.method_name = self.__class__

    def set_data(self, data):
        self.data = data
        self.subgraph = data.subgraph
        self.reset_centrality()

    def reset_centrality(self):
        centrality = {u: 0 for u in nx.nodes(self.subgraph)}
        nx.set_node_attributes(self.subgraph, centrality, 'centrality')

    def detect(self):
        return self.sort_nodes_by_centrality()

    def sort_nodes_by_centrality(self):
        result = nx.get_node_attributes(self.subgraph, 'centrality')
        result = sorted(result.items(), key=lambda d: d[1], reverse=True)
        return result


class DynamicMessagePassing(Method):
    """detect the source with DynamicMessagePassing.
        Please refer to the following paper for more details.
        Lokhov, Andrey Y., et al. "Inferring the origin of an epidemic with a dynamic message-passing algorithm."
        Physical Review E 90.1 (2014): 012801.
    """

    def detect(self):
        self.reset_centrality()
        nodes = self.data.graph.nodes()
        node2index = {}
        i = 0
        for v in nodes:
            node2index[v] = i
            i += 1
        nodes_infected = self.data.inode
        n = self.data.graph.number_of_nodes()
        n_i = self.subgraph.number_of_nodes()
        a = nx.adjacency_matrix(self.data.graph, weight=None).todense()  # adjacent matrix
        weights = np.asarray(nx.adjacency_matrix
                             (self.data.graph, weight='weight').todense())  # infection probability
        mu = np.zeros(n)  # Recovery rate, set to zero to consider SI
        try:  # The following line may raise nx.NetworkXError when the graph is not connected
            diameter = 1 * nx.diameter(self.data.graph)
        except nx.NetworkXError as msg:
            print(msg, "\n The Diameter is set to be the degree times 2.")
            diameter = self.data.order_origin * 2
        likelihoods = {}
        epsilon = 0.00001
        theta = np.ones([2, n, n])
        # theta[k][i] as the probability that the infection signal has not been passed
        phi = np.zeros([2, n, n])
        pij_s = np.zeros([2, n, n])
        p_sir = np.zeros([2, n_i, n, 3])  # time-source-node-state (SIR): probability
        ps0 = np.zeros([n, 3])  # the probability nodes i is s,i,r at time 0

        for s in np.arange(n_i):
            """initialize"""
            theta[0, :, :] = 1
            theta[1, :, :] = 0
            phi[1, :, :] = 0
            pij_s[0, :, :] = 0
            s_index = node2index[nodes_infected[s]]
            p_sir[0, s, :, 0] = ps0[:, 0] = 1  # S
            p_sir[0, s, :, 1] = ps0[:, 1] = 0  # I
            p_sir[0, s, s_index, 0] = ps0[s_index, 0] = 0
            p_sir[0, s, s_index, 1] = ps0[s_index, 1] = 1
            p_sir[0, s, :, 2] = ps0[:, 2] = 0
            phi[0] = np.repeat(ps0[:, 1], n).reshape(n, n)
            pij_s[0] = np.repeat(ps0[:, 0], n).reshape(n, n)

            """estimate the probabilities, P_s P_r P_I at time t"""
            likelihoods[s] = np.ones([diameter + 1])
            # Using list instead of dict to avoid hash problem
            t = 0
            while t < diameter:
                t += 1
                t_current = t % 2
                t_previous = (t - 1) % 2
                theta[t_current] = theta[t_previous] - np.multiply(weights, phi[t_previous])
                theta[t_current][np.where(np.abs(theta[t_current]) <= epsilon)] = epsilon
                p_sir[t_current, s, :, 0] = np.multiply(ps0[:, 0],
                                                        np.exp((np.dot(a, np.log(theta[t_current, :, :]))).diagonal()))
                p_sir[t_current, s, :, 2] = 0  # Only S and I states are considered
                p_sir[t_current, s, :, 1] = 1 - p_sir[t_current, s, :, 0] - p_sir[t_current, s, :, 2]
                for i in np.arange(n):
                    denominator = np.multiply(theta[t_current, :, 1], a[i, :])
                    denominator[np.where(denominator == 0)] = 1
                    pij_s[t_current, i] = np.divide(p_sir[t_current, s, i, 0], denominator)

                    phi[t_current] = np.multiply(np.multiply(1 - weights, 1 - mu),
                                                 phi[t_previous]) - (pij_s[t_current] - pij_s[t_previous])

                    """compute the likelihood by Eq. 21"""
                    for v in self.data.graph.nodes():
                        if v in self.subgraph.nodes():
                            likelihoods[s][t] *= p_sir[t_current, s, node2index[v], 1]
                        else:
                            likelihoods[s][t] *= p_sir[t_current, s, node2index[v], 0]
        """select t0 to maximizes the partition function Z(t) = \\sum_{node i}{P(o|i)}"""
        max_zt = -1
        t0 = 0
        for t in np.arange(1, diameter + 1):
            likelihoods_matrix = np.asarray(list(likelihoods.values()))
            zt = np.sum(likelihoods_matrix[:, t])
            if zt > max_zt:
                max_zt = zt
                t0 = t
        centrality = {}
        for v in np.arange(n_i):
            centrality[nodes_infected[v]] = likelihoods[v][t0]
        nx.set_node_attributes(self.subgraph, centrality, 'centrality')
        del theta, phi, pij_s, p_sir, ps0
        return self.sort_nodes_by_centrality()

    def detect_ori(self):
        self.reset_centrality()
        nodes_infected = list(self.subgraph.nodes())
        nodes = list(nodes_infected)
        for v in nodes_infected:
            neighbors = nx.all_neighbors(self.data.graph, v)
            for u in neighbors:
                if u not in nodes:
                    nodes.append(u)  # only include infected nodes and their neighbors
        graph_neighbor = nx.subgraph(self.data.graph, nodes)
        nodes = graph_neighbor.nodes()
        n = len(nodes)
        node2index = {}
        i = 0
        for v in nodes:
            node2index[v] = i
            i += 1

        n_i = self.subgraph.number_of_nodes()
        a = nx.adjacency_matrix(graph_neighbor, weight=None).todense()  # adjacent matrix
        weights = nx.adjacency_matrix(graph_neighbor, weight='weight').todense()  # infection probability
        mu = np.zeros(n)  # recover probability
        diameter = 2 * nx.diameter(self.subgraph)
        likelihoods = {}  # the likelihood, P(o|i), in Eq. 21.
        epsilon = 0.00001
        theta = np.ones([2, n, n])  # theta[k][i] as the probability that the infection signal has not been passed
        # from node k to node i up to time t in the dynamics Di
        phi = np.zeros([2, n, n])
        pij_s = np.zeros([2, n, n])
        p_sir = np.zeros([2, n_i, n, 3])  # time-source-node-state (SIR): probability
        ps0 = np.zeros([n, 3])  # the probability nodes i is s, i ,r at time 0

        for s in np.arange(n_i):
            """initialize"""
            theta[0, :, :] = 1
            theta[1, :, :] = 0
            phi[1, :, :] = 0
            pij_s[0, :, :] = 0
            s_index = node2index[nodes_infected[s]]
            p_sir[0, s, :, 0] = ps0[:, 0] = 1  # S
            p_sir[0, s, :, 1] = ps0[:, 1] = 0  # I
            p_sir[0, s, s_index, 0] = ps0[s_index, 0] = 0
            p_sir[0, s, s_index, 1] = ps0[s_index, 1] = 1
            p_sir[0, s, :, 2] = ps0[:, 2] = 0
            phi[0] = np.repeat(ps0[:, 1], n).reshape(n, n)
            pij_s[0] = np.repeat(ps0[:, 0], n).reshape(n, n)

            """estimate the probabilities, P_s P_r P_I at time t"""
            likelihoods[s] = np.ones([diameter + 1])
            t = 0
            while t < diameter:
                t += 1
                t_current = t % 2
                t_previous = (t - 1) % 2
                theta[t_current] = theta[t_previous] - np.multiply(weights, phi[t_previous])
                theta[t_current][np.where(np.abs(theta[t_current]) <= epsilon)] = epsilon

                p_sir[t_current, s, :, 0] = np.multiply(ps0[:, 0],
                                                        np.exp((np.dot(a, np.log(theta[t_current, :, :]))).diagonal()))
                p_sir[t_current, s, :, 2] = 0  # nodes only have two states: susceptible, infected
                p_sir[t_current, s, :, 1] = 1 - p_sir[t_current, s, :, 0] - p_sir[t_current, s, :, 2]
                for i in np.arange(n):
                    denominator = np.multiply(theta[t_current, :, i], a[i, :])
                    denominator[np.where(denominator == 0)] = 1
                    pij_s[t_current, i] = np.divide(p_sir[t_current, s, i, 0], denominator)

                phi[t_current] = np.multiply(np.multiply(1 - weights, 1 - mu), phi[t_previous]) - (
                        pij_s[t_current] - pij_s[t_previous])

                """compute the likelihood by Eq. 21"""
                for v in nodes:
                    if v in nodes_infected:
                        likelihoods[s][t] *= p_sir[t_current, s, node2index[v], 1]
                    else:
                        likelihoods[s][t] *= p_sir[t_current, s, node2index[v], 0]

        """select t0 to maximizes the partition function Z(t) = \sum_{node i}{P(o|i)}"""
        max_zt = -1
        t0 = 0
        for t in np.arange(1, diameter + 1):
            zt = np.sum(np.asarray(list(likelihoods.values()))[:, t])
            # print zt
            if zt > max_zt:
                max_zt = zt
                t0 = t
        centrality = {}
        for v in np.arange(n_i):
            centrality[nodes_infected[v]] = likelihoods[v][t0]
        nx.set_node_attributes(self.subgraph, centrality, 'centrality')
        return self.sort_nodes_by_centrality()


class ReverseInfection(Method):
    """detect the source with ReverseInfection.
        Please refer to the following paper for more details.
        Zhu K, Ying L. Information source detection in the SIR model: a sample-path-based approach[J].
        IEEE/ACM Transactions on Networking, 2016, 24(1): 408-421.
    """

    def detect(self):
        """detect the source with ReverseInfection.

        Returns:
            @rtype:int
            the detected source
        """
        self.reset_centrality()
        stop = 0
        t = 0
        n = len(nx.nodes(self.subgraph))
        ids_received = {}  # the ids and corresponding time one node has received
        ids_waiting = {}  # the ids and time one node will received in the next round

        """initialize"""
        for u in nx.nodes(self.subgraph):
            ids_received[u] = dict()
            ids_waiting[u] = dict()
            ids_waiting[u][u] = {'level': 0, 'time': 0}
            # for w in neighbors:
            #     ids_waiting[w][u] = t+1

        weight = nx.get_edge_attributes(self.subgraph, 'weight')
        while stop < n:
            # print 't=', t
            for u in nx.nodes(self.subgraph):
                for v in list(ids_waiting[u].keys()):
                    if ids_waiting[u][v]['level'] == t:
                        if v not in ids_received[u].keys():
                            ids_received[u][v] = ids_waiting[u][v]['time']
                            # u send v to its neighbours
                            neighbors = nx.all_neighbors(self.subgraph, u)
                            for w in neighbors:
                                if v not in ids_received[w].keys():
                                    if (u, w) in weight.keys():
                                        delay = weight[(u, w)]
                                    else:
                                        delay = weight[(w, u)]
                                    ids_waiting[w][v] = {'level': t + 1, 'time': ids_received[u][v] + delay}
                        ids_waiting[u].pop(v)
                # print ids_received[u], u
                # print ids_waiting[u]
                if len(ids_received[u]) == n:
                    nx.set_node_attributes(self.subgraph, {u: 1 / sum(ids_received[u].values())}, 'centrality')
                    stop += 1

            t += 1

        return self.sort_nodes_by_centrality()


class RumorCenter(Method):
    """
        detect the source with Rumor Centrality.
        Please refer to the following paper for more details.
        Shah D, Zaman T. Detecting sources of computer viruses in networks: theory and experiment[J].
        ACM SIGMETRICS Performance Evaluation Review, 2010, 38(1): 203-214.
    """

    visited = set()  # node set
    bfs_tree = nx.Graph()

    def detect(self):
        """detect the source with Rumor Centrality.

        Returns:
            @rtype:int
            the detected source
        """
        if self.subgraph.number_of_nodes() == 0:
            print("subgraph.number_of_nodes =0")
            return

        self.reset_centrality()
        centrality = {}
        for source in self.subgraph.nodes():
            self.bfs_tree = nx.bfs_tree(self.subgraph, source)
            self.visited.clear()
            self.get_number_in_subtree(source)
            centrality[source] = Decimal(math.factorial(self.bfs_tree.number_of_nodes())) \
                                 / nx.get_node_attributes(self.bfs_tree, 'cumulativeProductOfSubtrees')[source]

        nx.set_node_attributes(self.subgraph, centrality, 'centrality')
        return self.sort_nodes_by_centrality()

    def get_centrality(self, u):
        """get centralities for all nodes by passing a message from the root to the children.

        Args:
            u:
        """
        self.visited.add(u)
        centrality = 0
        if u == self.source:
            """p is the root node in the bfs_tree."""
            centrality = Decimal(math.factorial(self.bfs_tree.number_of_nodes())) \
                         / nx.get_node_attributes(self.bfs_tree, 'cumulativeProductOfSubtrees')[u]
        else:
            parent = nx.get_node_attributes(self.bfs_tree, 'parent')[u]
            numberOfNodesInSubtree = nx.get_node_attributes(self.bfs_tree, 'numberOfNodesInSubtree')[u]
            centrality = nx.get_node_attributes(self.bfs_tree, 'centrality')[parent] * numberOfNodesInSubtree / (
                    self.bfs_tree.number_of_nodes() - numberOfNodesInSubtree)
        centrality = centrality
        nx.set_node_attributes(self.bfs_tree, {u: centrality}, 'centrality')
        nx.set_node_attributes(self.subgraph, {u: centrality}, 'centrality')

        children = nx.all_neighbors(self.bfs_tree, u)
        for c in children:
            if c not in self.visited:
                self.get_centrality(c)

    def get_number_in_subtree(self, p):
        """passing messages from children nodes to the parent, to get the number of nodes in the subtree rooted by p,
        and the cumulative product of the size of the subtrees of all nodes in p' subtree.

        Args:
            p: parent node
        Returns:
            @rtype:Decimal()
        """
        self.visited.add(p)
        numberOfNodesInSubtree = 1  # for node p
        cumulativeProductOfSubtrees = Decimal(1)  # for node p

        children = nx.all_neighbors(self.bfs_tree, p)
        for u in children:
            if u not in self.visited:
                nx.set_node_attributes(self.bfs_tree, {u: p}, 'parent')
                self.get_number_in_subtree(u)
                numberOfNodesInSubtree += nx.get_node_attributes(self.bfs_tree, 'numberOfNodesInSubtree')[u]
                cumulativeProductOfSubtrees *= nx.get_node_attributes(self.bfs_tree, 'cumulativeProductOfSubtrees')[u]
        cumulativeProductOfSubtrees = Decimal(cumulativeProductOfSubtrees) * Decimal(numberOfNodesInSubtree)
        nx.set_node_attributes(self.bfs_tree, {p: numberOfNodesInSubtree}, 'numberOfNodesInSubtree')
        nx.set_node_attributes(self.bfs_tree, {p: cumulativeProductOfSubtrees}, 'cumulativeProductOfSubtrees')
