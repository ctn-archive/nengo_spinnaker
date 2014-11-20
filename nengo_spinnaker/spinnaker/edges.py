from pacman.model.partitionable_graph.partitionable_edge import \
    PartitionableEdge


class Edge(PartitionableEdge):
    """Edge with attendant keyspace."""
    def __init__(self, pre_vertex, post_vertex, keyspace):
        super(Edge, self).__init__(pre_vertex, post_vertex, "")
        self.keyspace = keyspace

    def __eq__(self, other):
        return all([
            self.__class__ is other.__class__,
            self.pre_vertex is other.pre_vertex,
            self.post_vertex is other.post_vertex,
            self.keyspace == other.keyspace,
        ])
