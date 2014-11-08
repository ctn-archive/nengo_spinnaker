class Edge(object):
    """Basic representation of an edge.

    Should be the minimum required by PACMAN, as the `create_subedge` never
    seems to be called anywhere meaningful.
    """
    __slots__ = ['pre_vertex', 'post_vertex', 'keyspace', 'label']

    def __init__(self, pre_vertex, post_vertex, keyspace):
        self.pre_vertex = pre_vertex
        self.post_vertex = post_vertex
        self.keyspace = keyspace
        self.label = ""  # Set this to something sensible later?

    def __eq__(self, other):
        return all([
            self.__class__ is other.__class__,
            self.pre_vertex is other.pre_vertex,
            self.post_vertex is other.post_vertex,
            self.keyspace == other.keyspace,
        ])
