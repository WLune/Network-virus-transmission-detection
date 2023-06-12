import copy
import pickle
import random

import networkx as nx


class InfectionGraph:
    def __init__(self, adjMatrix, status):
        """
        Initialize with adjacent matrix and status array provided

        :param adjMatrix: Adjacent matrix
            for the underlying graph
        :param status: The infection status of each node, in list format.
            0 for Susceptible, 1 for Infected, -1 for Unknown, -2 for Removed
        """
        self.graph = nx.Graph(adjMatrix)
        self.weights = nx.adjacency_matrix(self.graph, weight='weight')
        self.order_origin = self.graph.order()
        self.order = self.graph.order()
        self.status = list(status)
        self.snode = [i for i in range(0, self.order_origin) if self.status[i] == 0]
        self.inode = [i for i in range(0, self.order_origin) if self.status[i] == 1]
        self.unode = [i for i in range(0, self.order_origin) if self.status[i] == -1]
        self.subgraph = self.get_certain_subgraph().graph
        # This is the infected subgraph in nx graph object

    def update_attribute(self):
        self.order = self.graph.order()
        self.snode = [i for i in range(0, self.order_origin) if self.status[i] == 0]
        self.inode = [i for i in range(0, self.order_origin) if self.status[i] == 1]
        self.unode = [i for i in range(0, self.order_origin) if self.status[i] == -1]

    def remove_node(self, node):
        try:
            s = self.status[node]
        except IndexError:
            raise IndexError("Node %d is out of range" % node)
        if node not in self.graph.nodes():
            raise IndexError("Node %d is NOT in the graph" % node)
        self.status[node] = -2
        if s == 1:
            self.inode.remove(node)
        elif s == 0:
            self.snode.remove(node)
        elif s == -1:
            self.unode.remove(node)
        elif s == -2:
            # warnings.warn("Node %d has been removed before" % node, category=RuntimeWarning)
            print("Node %d has been removed before" % node)
        self.graph.remove_node(node)
        return

    def get_certain_subgraph(self, mode="leave", status=1):
        """
        Delete or leave a certain kind of node,
        by default, an infected subgraph is returned

        :param mode: "delete" or "leave",
            delete mode will remove the nodes in given state;
            leave mode will remove all of the nodes in the other states
        :param status: The certain kind of node to be deleted or left.
        :return:
            A InfectionGraph object
            containing the certain kinds of node

        Note:
        ----------
        The returned graph does not have to be connected

        To obtain the NetworkX graph object,
        use the "graph"attribute of the returned object

        """
        if mode not in ["delete", "leave"]:
            raise ValueError("Mode can only be \"delete\" or \"leave\"")
        graph_temp = copy.deepcopy(self)
        for node in range(graph_temp.order_origin):
            if mode == "delete":
                if graph_temp.status[node] == status:
                    graph_temp.remove_node(node)
            elif mode == "leave":
                if graph_temp.status[node] != status:
                    graph_temp.remove_node(node)
        return graph_temp

    def has_inode_path(self, source, target):
        """
        Apply Dijkstra algorithm to determine
        if path exists between source and target

        :param source:  An integer represents the source node
        :param target:  An integer represents the target node
        :return: True if path exists
        """
        if source not in self.inode or target not in self.inode:
            raise ValueError("Either source node %d or target node %d is NOT inode" % (source, target))
        temp = copy.deepcopy(self.graph)
        for node in range(self.order):
            if self.status[node] != 1:
                try:
                    temp.remove_node(node)
                except nx.exception.NetworkXError:
                    continue
        return nx.has_path(temp, source, target)

    def check_inode_connectivity(self):
        """

        :return: True if all of the inode are connected
        """
        if len(self.inode) == 0:
            raise ValueError("No inode found")
        if len(self.inode) == 1:
            return True
        for i in range(1, len(self.inode)):
            if not self.has_inode_path(self.inode[0], self.inode[i]):
                return False
        return True
		
class SourceGraph(InfectionGraph):
    def __init__(self, adjMatrix, status, path=' ', comments='#', weighted=0):
        """
        Initialize with edge list in file and status array provided

        :param status: The infection status of each node, in list format.
            0 for Susceptible, 1 for Infected, -1 for Unknown, -2 for Removed
        :param path:  The file path of edge list
        :param comments: Marker for comment lines. Default is `'#'`
        :param weighted: 0 for random weight, 1 for  weight assigned in file
        """
        super().__init__(adjMatrix, status)
        self.node2index = {}
        self.index2node = {}
        if weighted == 0:
            if path.endswith('.gml'):
                self.graph = nx.read_gml(path)
            elif path.endswith('.txt'):
                self.graph = nx.read_edgelist(path, comments=comments)
            self.set_weight_random()
        elif weighted == 1:
            if path.endswith('.gml'):
                self.graph = nx.read_gml(path)
            elif path.endswith('.txt'):
                self.graph = nx.read_weighted_edgelist(path, comments=comments)
        self.weights = nx.adjacency_matrix(self.graph, weight='weight')
        i = 0
        for v in self.graph.nodes():
            self.node2index[v] = i
            self.index2node[i] = v
            i += 1
        self.ratio_infected = 0

    def set_weight_random(self):
        a = {e: random.random() for e in self.graph.edges()}
        nx.set_edge_attributes(self.graph, a, 'weight')

    def infect_from_source_SI(self, source, scheme='random', infected_size=None):
        max_infected_number = self.ratio_infected * self.graph.number_of_nodes()
        if infected_size is not None:
            max_infected_number = infected_size
        infected = set()
        waiting = set()
        infected.add(source)
        waiting.add(source)
        stop = False
        if scheme == 'random':
            while stop is False and (waiting.__len__() < max_infected_number) and (
                    waiting.__len__() < self.graph.number_of_nodes()):
                for w in waiting:
                    if stop:
                        break
                    neighbors = nx.all_neighbors(self.graph, w)
                    for u in neighbors:
                        if u not in infected:
                            weight = self.weights[self.node2index[w], self.node2index[u]]
                            if random.random() <= weight:
                                infected.add(u)
                        if len(infected) >= max_infected_number:
                            stop = True
                            break
            waiting = infected.copy()
        elif scheme == 'snowball':
            while (waiting.__len__() <= max_infected_number) and (waiting.__len__() <= self.graph.number_of_nodes()):
                for w in neighbors:
                    neighbors = nx.all_neighbors(self.graph, w)
                    for u in neighbors:
                        if u not in infected:
                            infected.add(u)
                waiting = infected
        self.subgraph = self.graph.subgraph(infected)
        return infected

    def generate_infected_subgraph(self, output_file_prefix, ratio_infected, scheme='random'):
        self.ratio_infected = ratio_infected
        i = 0
        for v in self.graph.nodes():
            self.infect_from_source_SI(v, scheme=scheme)
            """prefix+numberOfInfectedNodes+source"""
            output_file = "%s.i%s.s%s.subgraph" % (output_file_prefix, self.subgraph.number_of_nodes(), i)
            print(output_file, self.subgraph.number_of_nodes())
            writer = open(output_file, "w")
            pickle.dump(self, writer)
            writer.close()
            i += 1

    def get_diameter_for_subgraphs(self, infected_size, scheme='random'):
        """the longest shortest road"""
        diameter = 0.0
        ratio_edge2node = 0.0
        for v in self.graph.nodes():
            self.infect_from_source_SI(v, scheme=scheme, infected_size=infected_size)
            diameter += nx.diameter(self.subgraph)
            ratio_edge2node += self.subgraph.number_of_edges() * 1.0 / self.subgraph.number_of_nodes()
        return diameter / self.graph.number_of_nodes(), ratio_edge2node / self.graph.number_of_nodes()
