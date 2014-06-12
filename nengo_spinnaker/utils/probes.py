import numpy as np
import logging
import warnings
logger = logging.getLogger(__name__)

import nengo
import nengo.utils.builder

from . import vertices, filters
from . import fixpoint as fp
from .. import edges


def get_probe_nodes_connections(probes):
    """Return a list of the ProbeNodes and Connections which need to be added
    to Probe the output of selected Nodes."""
    new_nodes = list()
    new_conns = list()

    for p in probes:
        if isinstance(p.target, nengo.Node):
            # Create a new ProbeNode, Connection
            pn = ProbeNode(p, add_to_container=False)
            pc = nengo.Connection(p.target, pn,
                                  synapse=p.conn_args.get('synapse', None),
                                  add_to_container=False)

            new_nodes.append(pn)
            new_conns.append(pc)

    return new_nodes, new_conns


def get_corrected_probes(probes, connections):
    """Get a modified list of Probes, removing the synapses from Probes which
    probe PassNodes with synapses on *their* inputs.
    """
    new_probes = list()
    ins, _ = nengo.utils.builder.find_all_io(connections)

    for probe in probes:
        if (isinstance(probe.target, nengo.Node) and
                probe.target.output is None):
            # This is a PassNode, modify if required
            # Get the incoming connections to the target
            if True in [c.synapse is not None for c in ins[probe.target]]:
                warnings.warn("Can't currently have synapse on PassNode Probe."
                              "\n\nDefaulting to synapse=None on %s\n"
                              "This is because you tried to provide a"
                              "synapse for a PassNode Probe where inputs to"
                              "the PassNode already possessed synapses.\n"
                              % probe, RuntimeWarning)
                probe = nengo.Probe(probe.target, synapse=None,
                                    add_to_container=False)
        new_probes.append(probe)
    return new_probes


def build_ensemble_probenode_edge(builder, c):
    prevertex = builder.ensemble_vertices[c.pre]
    edge = edges.DecoderEdge(c, prevertex, c.post.vertex)
    return edge


def build_node_probenode_edge(builder, c):
    prevertex = builder.get_node_out_vertex(c)
    edge = edges.NengoEdge(c, prevertex, c.post.vertex,
                           filter_is_accumulatory=False)
    return edge


class SpiNNakerProbe(object):
    """A NengoProbe encapsulates the logic required to retrieve data from a
    SpiNNaker machine.
    """
    def __init__(self, probe, dt=0.001):
        self.probe = probe
        self.dt = dt

    def get_data(self, txrx):
        raise NotImplementedError


class ProbeNode(nengo.Node):
    def __init__(self, probe, output=None):
        self.output = lambda t: t  # NOT None!
        self.probe = probe
        self.size_in = probe.target.size_out
        self.vertex = None

    def spinnaker_build(self, builder):
        """Add a new ValueSinkVertex and Probe to the Builder."""
        self.vertex = ValueSinkVertex(self.size_in)
        builder.add_vertex(self.vertex)
        builder.probes.append(DecodedValueProbe(self.vertex, self.probe))


class DecodedValueProbe(SpiNNakerProbe):
    def __init__(self, recording_vertex, probe):
        super(DecodedValueProbe, self).__init__(probe)
        self.recording_vertex = recording_vertex

    def get_data(self, txrx):
        # For only 1 subvertex, get the recorded data
        assert(len(self.recording_vertex.subvertices) == 1)
        sv = self.recording_vertex.subvertices[0]
        (x, y, p) = sv.placement.processor.get_coordinates()

        sdata = vertices.retrieve_region_data(
            txrx, x, y, p, self.recording_vertex.REGIONS['VALUES'],
            self.recording_vertex.sizeof_values(sv.n_atoms)
        )

        # Cast as a Numpy array, shape and return
        data = np.array(fp.kbits([int(i) for i in
                                  np.fromstring(sdata, dtype=np.uint32)]))
        return data.reshape((self.recording_vertex.run_ticks,
                             self.recording_vertex.width))


try:
    from bitarray import bitarray

    class SpikeProbe(SpiNNakerProbe):
        def __init__(self, target_vertex, probe):
            super(SpikeProbe, self).__init__(probe)
            self.target_vertex = target_vertex

        def get_data(self, txrx):
            # Calculate the number of frames
            n_frames = int(self.target_vertex.runtime * 1000)  # TODO Neaten!
            data = [list() for n in range(n_frames)]

            for subvertex in self.target_vertex.subvertices:
                # Get the contents of the "SPIKES" region for each subvertex
                (x, y, p) = subvertex.placement.processor.get_coordinates()

                sdata = vertices.retrieve_region_data(
                    txrx, x, y, p, self.target_vertex.REGIONS["SPIKES"],
                    self.target_vertex.sizeof_region_recording(
                        subvertex.n_atoms)
                )

                # Cast the spikes as a bit array
                spikes = bitarray()
                spikes.frombytes(sdata)

                # Convert each frame into a list of spiked neurons
                frame_length = ((subvertex.n_atoms >> 5) +
                                (1 if subvertex.n_atoms & 0x1f else 0))

                for f in range(n_frames):
                    frame = spikes[32*f*frame_length + 32:
                                   32*(f + 1)*frame_length + 32]
                    data[f].extend([n + subvertex.lo_atom for n in
                                    range(subvertex.n_atoms) if frame[n]])

            # Convert into list of spike times
            spikes = [[0.] for n in range(self.probe.target.n_neurons)]
            for (i, f) in enumerate(data):
                for n in f:
                    spikes[n].append(i*self.dt)

            return spikes

except ImportError:
    # No bitarray, so no spike probing!
    warnings.warn("Cannot import module bitarray: spike probing is disabled",
                  ImportWarning)
    SpikeProbe = None


@filters.with_filters(2, 3)
class ValueSinkVertex(vertices.NengoVertex):
    """ValueSinkVertex records the decoded values which it receives.
    """
    REGIONS = vertices.ordered_regions('SYSTEM', **{'VALUES': 15})
    MODEL_NAME = "nengo_value_sink"

    def __init__(self, width, dt=0.001, timestep=1000):
        super(ValueSinkVertex, self).__init__(1, constraints=None,
                                              label="Nengo Value Sink")
        self.width = width
        self.timestep = timestep
        self.dt = dt

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    def dtcm_usage(self, n_atoms):
        return self.sizeof_system(n_atoms) * 4

    @vertices.region_pre_prepare('SYSTEM')
    def prepare_system(self):
        # Calculate the number of ticks of execution
        self.run_ticks = ((1 << 32) - 1 if self.runtime is None else
                          int(self.runtime * 1000))

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_system(self, n_atoms):
        # 1. Timestep
        # 2. Width
        return 2

    @vertices.region_pre_sizeof('VALUES')
    def sizeof_values(self, n_atoms):
        if self.runtime is None:
            logger.warning("Can't record for an indefinite period, "
                           "not recording at all.")
            return 0
        else:
            return self.width * self.run_ticks

    @vertices.region_write('SYSTEM')
    def write_system(self, subvertex, spec):
        spec.write(data=self.timestep)
        spec.write(data=self.width)
