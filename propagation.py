# -*- coding:utf-8-*-
# SIR Propagation Process
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


class SIRPropagation:
    def __init__(self, N, N0, irate, rrate, graph_type="small-world"):
        """

        :param N: Total number of nodes
        :param N0:  Number of patient(s) zero
        :param irate: Infection rate
        :param rrate: Recovery rate
        :param graph_type: The type of underlying graph
        """
        self.number_of_nodes = N
        self.number_of_patient0 = N0
        self.infection_rate = irate
        self.recovery_rate = rrate
        self.graph_type = graph_type
        self.adj_matrix = np.array([])
        self.status = self.__patient_zero()
        self.time_status = {}

    def generate_small_world(self, deg, a):
        """

        :param deg: degree of nodes
        :param a: probability that nodes connected to nodes other than adjacent number
        :return: the adjacent matrix
        """
        random.seed(1024)
        A = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for i in range(self.number_of_nodes):
            t = 0
            while t < (deg / 2):
                A[i][i - (t + 1)] = 1
                A[i - (t + 1)][i] = 1
                t += 1

        for i in range(self.number_of_nodes):
            t = 0
            while t < (self.number_of_nodes / 2):
                if A[i][i - (t + 1)] == 1:
                    if random.random() < a:
                        A[i][i - (t + 1)] = 0
                        A[i - (t + 1)][i] = 0
                        target = random.randint(0, (self.number_of_nodes - 1))
                        while A[i][target] == 1 or target == i:
                            target = random.randint(0, (self.number_of_nodes - 1))
                        A[i][target] = 1
                        A[target][i] = 1
                t += 1
        self.adj_matrix = A
        return A

    def generate_barabasi_albert_graph(self, n_init, n_new):
        """
        Generate a BA graph

        :param n_init: The initial number of nodes that connected
        :param n_new: The number of nodes that each new node will be connected to.
        :return: The adjacent matrix
        """
        assert (n_init < self.number_of_nodes)
        assert (n_new <= n_init)

        # adjacency matrix
        A = np.zeros((self.number_of_nodes, self.number_of_nodes))

        for i in range(0, n_init):
            for j in range(i + 1, n_init):
                A[i, j] = 1
                A[j, i] = 1

        # add 'c' node
        for c in range(n_init, self.number_of_nodes):
            Allk = np.sum(A)  # all  degree    Eki
            ki = np.sum(A, axis=1)  # ki each degree for node i

            pi = np.zeros(c, dtype=np.float)  # probability
            for i in range(0, c):
                pi[i] = ki[i] / (Allk * 1.0)
            # print pi

            # connect m edges.
            for d in range(0, n_new):
                rand01 = random.random()  # [0,1.0)

                sumpi = 0.0
                for g in range(0, c):
                    sumpi += pi[g]
                    if sumpi > rand01:  # connect 'c' node with 'g' node.
                        A[c, g] = 1
                        A[g, c] = 1
                        break
        self.adj_matrix = A
        return A

    def __patient_zero(self):
        """
        Initialize the patient(s) zero

        :return: The status list in which 0 for susceptible, 1 for infectious
        """
        patients = random.sample(range(self.number_of_nodes), self.number_of_patient0)
        self.patient0 = patients
        self.status = np.zeros(self.number_of_nodes, int)
        for i in patients:
            self.status[i] = 1
        return self.status

    def __infect(self):
        """
        The infection process

        :return: The status list
        """
        N = len(self.adj_matrix)
        for i in range(N):
            if self.status[i] == 1 and random.random() <= self.recovery_rate:
                self.status[i] = 2

        if sum(self.status == 1) < N / 2:
            for i in range(N):
                if self.status[i] == 1:
                    for j in range(N):
                        if self.adj_matrix[i][j] == 1 and \
                                self.status[j] == 0 and \
                                random.random() <= self.infection_rate:
                            self.status[j] = 1
        else:
            for i in range(N):
                if self.status[i] == 0:
                    for j in range(N):
                        if self.adj_matrix[i][j] == 1 and \
                                self.status[j] == 1 and \
                                random.random() <= self.infection_rate:
                            self.status[i] = 1
        return self.status

    def SIR(self, plot=False):
        N = len(self.adj_matrix)

        if plot:
            g = nx.from_numpy_matrix(self.adj_matrix)
            pos = nx.kamada_kawai_layout(g)
            nodesize = []
            maxsize = 10
            minsize = 1
            maxdegree = np.max(np.sum(self.adj_matrix, axis=0))
            mindegree = np.min(np.sum(self.adj_matrix, axis=0))
            if maxdegree == mindegree:
                nodesize = [minsize for i in range(len(self.adj_matrix))]
            else:
                for node in g:
                    size = (np.sum(self.adj_matrix[node]) - mindegree) / \
                           (maxdegree - mindegree) * (maxsize - minsize) + minsize
                    nodesize.append(size)

        result = []
        time = 0
        print_flag = True
        dup_cnt = 0
        while True:
            if plot:
                cmap = ['g', 'r', 'b']
                colors = [cmap[s] for s in self.status]
                plt.figure(figsize=(20, 20))
                if time == 0:
                    nx.draw(g, pos=pos, with_labels=True, node_color=colors, alpha=0.6)
                nx.draw_networkx_nodes(g, pos=pos, node_color=colors, alpha=0.6)
                nx.draw_networkx_edges(g, pos=pos, width=0.3, alpha=0.3)
                plt.title('time = {}'.format(time))
                # plt.savefig('{}.png'.format(str(time).zfill(4)))
                plt.show()
                plt.pause(0.1)

            result.append((sum(self.status == 0), sum(self.status == 1), sum(self.status == 2)))
            if time >= 2 and result[-1] == result[-2]:
                dup_cnt += 1
            else:
                dup_cnt = 0

            if sum(self.status == 1) == N or sum(self.status == 1) == 0 or time > N ** 3 or dup_cnt > N // 10:
                break
            self.time_status[time] = self.status.tolist()
            self.status = self.__infect()
            time += 1

        return np.array(result)

    def plot_trends(self):
        result = []
        for t in self.time_status.keys():
            result.append((sum(np.array(self.time_status[t]) == 0),
                           sum(np.array(self.time_status[t]) == 1),
                           sum(np.array(self.time_status[t]) == 2)))
        result = np.array(result)
        plt.plot(result[:, 0], 'g', label='Susceptibles')
        plt.plot(result[:, 1], 'r', label='Infectious')
        plt.plot(result[:, 2], 'b', label='Recovereds')
        plt.legend()