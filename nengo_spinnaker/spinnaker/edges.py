import collections


class Edge(collections.namedtuple('Edge', 'pre_vertex post_vertex keyspace')):
    """Represents a communication channel on a SpiNNaker machine."""
