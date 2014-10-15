import collections
import itertools

import nengo

from .config import Config
from .spinnaker import vertices
# import connection
# import ensemble
# import node
# import probe
# import utils


class Assembler(object):
    """The Assembler object takes a built collection of objects and connections
    and converts them into PACMAN vertices and edges, and returns the portion
    of the network to be simulated on host.
    """
    object_builders = dict()  # Map of classes to functions

    # Map of (pre_obj, post_obj) tuples to functions
    connection_builders = dict()

    @classmethod
    def register_object_builder(cls, func, nengo_class):
        cls.object_builders[nengo_class] = func

    @classmethod
    def register_connection_builder(cls, func, pre_obj=None, post_obj=None):
        cls.connection_builders[(pre_obj, post_obj)] = func

    def build_object(self, obj):
        for obj_type in obj.__class__.__mro__:
            if obj_type in self.object_builders:
                break
        else:
            raise TypeError("Cannot assemble object of type '%s'." %
                            obj.__class__.__name__)

        vertex = self.object_builders[obj_type](obj, self)
        if vertex is not None:
            vertex.runtime = self.time_in_seconds
        return vertex

    def build_connection(self, connection):
        pre_c = list(connection.pre_obj.__class__.__mro__) + [None]
        post_c = list(connection.post_obj.__class__.__mro__) + [None]

        for key in itertools.chain(*[[(a, b) for b in post_c] for a in pre_c]):
            if key in self.connection_builders:
                return self.connection_builders[key](connection, self)
        else:
            raise TypeError("Cannot build a connection from a '%s' to '%s'." %
                            (connection.pre_obj.__class__.__name__,
                             connection.post_obj.__class__.__name__))

    def __call__(self, objs, conns, time_in_seconds, dt, config=None):
        """Construct PACMAN vertices and edges, and a reduced version of the
        model for simulation on host.

        :param objs: A list of objects to convert into PACMAN vertices.
        :param conns: A list of connections which will become edges.
        :param time_in_seconds: The run time of the simulation (None means
                                infinite).
        :param dt: The time step of the simulation.
        :param config: Configuration options for the simulation.
        """
        # Store the config
        self.config = config
        if self.config is None:
            self.config = Config()

        self.timestep = 1000
        self.dt = dt
        self.time_in_seconds = time_in_seconds
        self.n_ticks = (int(time_in_seconds / dt) if
                        time_in_seconds is not None else 0)

        # Store for querying
        self.connections = conns

        # Construct each object in turn to produce vertices
        self.object_vertices = dict([(o, self.build_object(o)) for o in objs])
        self.vertices = [v for v in self.object_vertices.values() if
                         v is not None]

        # Construct each connection in turn to produce edges
        self.edges = filter(lambda x: x is not None, [self.build_connection(c)
                                                      for c in conns])

        return self.vertices, self.edges

    def get_object_vertex(self, obj):
        """Return the vertex which represents the given object."""
        return self.object_vertices[obj]

    def get_incoming_connections(self, obj):
        return [c for c in self.connections if c.post_obj == obj]

    def get_outgoing_connections(self, obj):
        return [c for c in self.connections if c.pre_obj == obj]


def vertex_builder(vertex, assembler):
    return vertex

Assembler.register_object_builder(vertex_builder, vertices.NengoVertex)


def assemble_node(node, assembler):
    pass

Assembler.register_object_builder(assemble_node, nengo.Node)


MulticastPacket = collections.namedtuple('MulticastPacket',
                                         ['timestamp', 'key', 'payload'])


"""
class MulticastPlayer(utils.vertices.NengoVertex):
    # NOTE This is intended to be temporary while PACMAN is refactored
    MODEL_NAME = 'nengo_mc_player'
    MAX_ATOMS = 1

    def __init__(self):
        super(MulticastPlayer, self).__init__(1)
        self.regions = [None, None, None, None]

    @classmethod
    def assemble(cls, mcp, assembler):
        # Get all the symbols to transmit prior to and after the simulation
        sinks = set(
            c.post_obj for c in assembler.get_outgoing_connections(mcp))

        start_items = list()
        end_items = list()

        for sink in sinks:
            for p in sink.start_packets:
                start_items.extend([0, p.key,
                                    0 if p.payload is None else p.payload,
                                    p.payload is not None])

            for p in sink.end_packets:
                end_items.extend([0, p.key,
                                  0 if p.payload is None else p.payload,
                                  p.payload is not None])

        # Build the regions
        start_items.insert(0, len(start_items)/4)
        start_region = utils.vertices.UnpartitionedListRegion(
            start_items)
        end_items.insert(0, len(end_items)/4)
        end_region = utils.vertices.UnpartitionedListRegion(
            end_items)
        mcp.regions[1] = start_region
        mcp.regions[3] = end_region

        return mcp

Assembler.register_object_builder(MulticastPlayer.assemble, MulticastPlayer)
"""
