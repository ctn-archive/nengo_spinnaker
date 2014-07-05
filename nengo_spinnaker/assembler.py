import collections
import itertools
import numpy as np

import nengo

import pacman103
from .config import Config
from . import builder
from . import utils


class Assembler(object):
    """The Assembler object takes a built collection of objects and connections
    and converts them into PACMAN vertices and edges, and returns the portion
    of the network to be simulated on host.
    """
    object_builders = dict()  # Map of classes to functions
    connection_builders = dict()  # Map of (pre, post) tuples to functions

    @classmethod
    def register_object_builder(cls, func, nengo_class):
        cls.object_builders[nengo_class] = func

    @classmethod
    def register_connection_builder(cls, func, pre=None, post=None):
        cls.connection_builders[(pre, post)] = func

    def build_object(self, obj):
        for obj_type in obj.__class__.__mro__:
            if obj_type in self.object_builders:
                break
        else:
            raise TypeError("Cannot assemble object of type '%s'." %
                            obj.__class__.__name__)

        vertex = self.object_builders[obj_type](obj, self)
        if vertex is not None:
            assert isinstance(vertex, pacman103.lib.graph.Vertex)
            vertex.runtime = self.time_in_seconds
        return vertex

    def build_connection(self, connection):
        pre_c = list(connection.pre.__class__.__mro__) + [None]
        post_c = list(connection.post.__class__.__mro__) + [None]

        for key in itertools.chain(*[[(a, b) for b in post_c] for a in pre_c]):
            if key in self.connection_builders:
                return self.connection_builders[key](connection, self)
        else:
            raise TypeError("Cannot build a connection from a '%s' to '%s'." %
                            (connection.pre.__class__.__name__,
                             connection.post.__class__.__name__))

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
        self.n_ticks = int(time_in_seconds / dt)

        # Store for querying
        self.connections = conns

        # Construct each object in turn to produce vertices
        self.object_vertices = dict([(o, self.build_object(o)) for o in objs])
        self.vertices = [v for v in self.object_vertices.values() if
                         v is not None]

        # Construct each connection in turn to produce edges
        self.edges = [self.build_connection(c) for c in conns]

        return self.vertices, self.edges

    def get_object_vertex(self, obj):
        """Return the vertex which represents the given object."""
        return self.object_vertices[obj]

    def get_incoming_connections(self, obj):
        return [c for c in self.connections if c.post == obj]

    def get_outgoing_connections(self, obj):
        return [c for c in self.connections if c.pre == obj]


class NengoEdge(pacman103.lib.graph.Edge):
    def __init__(self, prevertex, postvertex, keyspace):
        super(NengoEdge, self).__init__(prevertex, postvertex)
        self.keyspace = keyspace


def generic_connection_builder(connection, assembler):
    """Builder for connections which just require an edge between two vertices
    """
    # Get the pre and post objects
    prevertex = assembler.get_object_vertex(connection.pre)
    postvertex = assembler.get_object_vertex(connection.post)

    # Ensure that the keyspace is set
    assert connection.keyspace is not None

    # Create and return the edge
    return NengoEdge(prevertex, postvertex, connection.keyspace)

Assembler.register_connection_builder(generic_connection_builder)


def vertex_builder(vertex, assembler):
    return vertex

Assembler.register_object_builder(vertex_builder, pacman103.lib.graph.Vertex)


class EnsembleLIF(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_ensemble'
    MAX_ATOMS = 128
    spikes_recording_region = 15

    def __init__(self, n_neurons, system_region, bias_region, encoders_region,
                 decoders_region, output_keys_region, input_filter_region,
                 input_filter_routing, inhib_filter_region,
                 inhib_filter_routing, gain_region, spikes_region):
        super(EnsembleLIF, self).__init__(n_neurons)
        # Create regions
        self.regions = [None]*16
        self.regions[0] = system_region
        self.regions[1] = bias_region
        self.regions[2] = encoders_region
        self.regions[3] = decoders_region
        self.regions[4] = output_keys_region
        self.regions[5] = input_filter_region
        self.regions[6] = input_filter_routing
        self.regions[7] = inhib_filter_region
        self.regions[8] = inhib_filter_routing
        self.regions[9] = gain_region
        self.regions[14] = spikes_region
        self.probes = list()

    @classmethod
    def assemble(cls, ens, assembler):
        # Prepare the system region
        system_items = [
            ens.n_dimensions,
            len(ens.output_keys),
            ens.n_neurons,
            assembler.timestep,
            int(ens.tau_ref / (assembler.timestep * 10**-6)),
            utils.fp.bitsk(assembler.dt / ens.tau_rc),
            0x1 if ens.record_spikes else 0x0,
            1
        ]

        # Prepare the input filtering regions
        # Prepare the inhibitory filtering regions
        in_conns = assembler.get_incoming_connections(ens)
        inhib_conns =\
            [c for c in in_conns if
             isinstance(c, builder.IntermediateGlobalInhibitionConnection)]
        input_conns = [c for c in in_conns if c not in inhib_conns]

        (input_filter_region, input_filter_routing) =\
            utils.vertices.make_filter_regions(input_conns, assembler.dt)
        (inhib_filter_region, inhib_filter_routing) =\
            utils.vertices.make_filter_regions(inhib_conns, assembler.dt)

        # Generate all the regions in turn, then return a new vertex
        # instance.
        encoders_with_gain = ens.encoders * ens.gains[:, np.newaxis]
        bias_with_di = np.dot(encoders_with_gain, ens.direct_input) + ens.bias

        system_region = utils.vertices.UnpartitionedListRegion(
            system_items, n_atoms_index=2)
        bias_region = utils.vertices.MatrixRegionPartitionedByRows(
            bias_with_di, formatter=utils.fp.bitsk)
        encoders_region = utils.vertices.MatrixRegionPartitionedByRows(
            encoders_with_gain, formatter=utils.fp.bitsk)
        decoders_region = utils.vertices.MatrixRegionPartitionedByRows(
            ens.decoders, formatter=utils.fp.bitsk)
        output_keys_region = utils.vertices.UnpartitionedListRegion(
            ens.output_keys)
        gain_region = utils.vertices.MatrixRegionPartitionedByRows(
            ens.gains, formatter=utils.fp.bitsk)
        spikes_region = utils.vertices.BitfieldBasedRecordingRegion(
            assembler.n_ticks)

        vertex = cls(ens.n_neurons, system_region, bias_region,
                     encoders_region, decoders_region, output_keys_region,
                     input_filter_region, input_filter_routing,
                     inhib_filter_region, inhib_filter_routing, gain_region,
                     spikes_region)
        vertex.probes = ens.probes
        return vertex

Assembler.register_object_builder(EnsembleLIF.assemble,
                                  builder.IntermediateLIFEnsemble)


def assemble_node(node, assembler):
    pass

Assembler.register_object_builder(assemble_node, nengo.Node)


class DecodedValueProbe(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_value_sink'
    MAX_ATOMS = 1
    recording_region_index = 15

    def __init__(self, system_region, input_filter_region,
                 input_filter_routing, recording_region, probe):
        super(DecodedValueProbe, self).__init__(1)
        self.regions = [None]*16
        self.regions[0] = system_region
        self.regions[1] = input_filter_region
        self.regions[2] = input_filter_routing
        self.regions[14] = recording_region
        self.probe = probe
        self.width = probe.size_in

    @classmethod
    def assemble(cls, probe, assembler):
        system_items = [assembler.timestep, probe.size_in]
        system_region = utils.vertices.UnpartitionedListRegion(system_items)

        if assembler.time_in_seconds is None:
            raise NotImplementedError

        # Build the input filters
        in_conns = assembler.get_incoming_connections(probe)
        (input_filter_region, input_filter_routing) =\
            utils.vertices.make_filter_regions(in_conns, assembler.dt)

        # Prepare the recording region
        recording_region = utils.vertices.FrameBasedRecordingRegion(
            probe.size_in, assembler.n_ticks)

        return cls(system_region, input_filter_region, input_filter_routing,
                   recording_region, probe.probe)

Assembler.register_object_builder(DecodedValueProbe.assemble,
                                  builder.IntermediateProbe)


class ValueSource(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_value_source'
    MAX_ATOMS = 1

    def __init__(self, system_region, keys_region, data_region):
        super(ValueSource, self).__init__(1)
        self.regions = [system_region, keys_region, data_region]

    @classmethod
    def from_node(cls, fn, conns, duration, periodic, dt):
        # Generate some evaluation points, construct the signal for the given
        # duration.
        ts = np.arange(0, duration, dt)
        data = []
        for t in ts:
            v = np.array(fn(t))

            output = []
            for tf in conns.transforms_functions:
                output.append(np.asarray(
                    np.dot(tf.transform, v if tf.function is None
                           else tf.function(v))))

            data.append(np.vstack(output))
        data = np.array(data)
        data.shape = (1, data.size)

        # Calculate the number of blocks
        frames_per_block = 5*1024/conns.width
        full_blocks = int(duration / dt) / frames_per_block
        partial_block = int(duration / dt) % frames_per_block

        # Prepare the system region, etc.
        system_items = [
            1000,
            conns.width,
            0x1 if periodic else 0x0,
            full_blocks,
            frames_per_block,
            partial_block
        ]
        system_region = utils.vertices.UnpartitionedListRegion(system_items)

        output_keys = utils.connections.get_output_keys(conns)
        output_keys_region = utils.vertices.UnpartitionedListRegion(
            output_keys)

        data_region = utils.vertices.MatrixRegionPartitionedByRows(
            data, in_dtcm=False, formatter=utils.fp.bitsk)

        return cls(system_region, output_keys_region, data_region)


class FilterVertex(utils.vertices.NengoVertex):
    MODEL_NAME = 'nengo_filter'
    MAX_ATOMS = 1

    def __init__(self, size_in, in_connections, dt, output_period=100,
                 interpacket_pause=1):
        super(FilterVertex, self).__init__(1)
        self.size_in = size_in

        # Create the system region
        system_region = utils.vertices.UnpartitionedListRegion([
            size_in, 1000, output_period, interpacket_pause])

        # Create the filter regions
        (in_filters, in_routing) = utils.vertices.make_filter_regions(
            in_connections, dt)
        self.regions = [system_region, None, in_filters, in_routing]

    @classmethod
    def get_output_keys_region(cls, fv, assembler):
        # Add output keys to the given fv component
        out_conns = assembler.get_outgoing_connections(fv)
        assert len(out_conns) == 0 or len(out_conns) == 1  # Only one out edge

        output_keys = list()

        for c in out_conns:
            for d in range(fv.size_in):
                output_keys.append(c.keyspace.key(d=d))

        return utils.vertices.UnpartitionedListRegion(output_keys)

    @classmethod
    def assemble(cls, fv, assembler):
        # Create the output keys region and add it to the instance, then
        # return.
        fv.regions[1] = cls.get_output_keys_region(fv, assembler)
        return fv

    @classmethod
    def assemble_from_intermediate(cls, fv, assembler):
        # Create the vertex, then assemble that and return
        in_conns = utils.connections.Connections(
            assembler.get_incoming_connections(fv))

        fv_ = cls(fv.size_in, in_conns, assembler.dt)
        fv_.regions[1] = cls.get_output_keys_region(fv, assembler)

        return fv_

Assembler.register_object_builder(FilterVertex.assemble, FilterVertex)
Assembler.register_object_builder(FilterVertex.assemble_from_intermediate,
                                  builder.IntermediateFilter)


MulticastPacket = collections.namedtuple('MulticastPacket',
                                         ['timestamp', 'key', 'payload'])


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
        sinks = [c.post for c in assembler.get_outgoing_connections(mcp)]

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
