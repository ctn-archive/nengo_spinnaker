import utils


def insert_decoded_output_probes(objs, connections, probes):
    """Creates a new object representing decoded output probes and provides
    appropriate connections.
    """
    objs = list(objs)
    connections = list(connections)

    # Add new objects and connections for 'decoded output' probes
    for probe in probes:
        if probe.attr == 'decoded_output' or probe.attr == 'output':
            p = IntermediateProbe(probe.size_in, probe.sample_every, probe,
                                  probe.label)

            # Create a new connection for this Node, if there is no transform
            # on the connection then we can create one on the assumption that
            # size_in and size_out are equivalent.
            c = utils.builder.IntermediateConnection(
                probe.target, p, synapse=probe.synapse, solver=probe.solver)

            # Add the new probe object and connection object
            objs.append(p)
            connections.append(c)

    return objs, connections


class IntermediateProbe(object):
    def __init__(self, size_in, sample_every, probe, label=None):
        self.size_in = size_in
        self.sample_every = sample_every
        self.probe = probe
        self.label = label


"""
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
            return None

        # Build the input filters
        in_conns = assembler.get_incoming_connections(probe)
        (input_filter_region, input_filter_routing, _) =\
            utils.vertices.make_filter_regions(in_conns, assembler.dt)

        # Prepare the recording region
        recording_region = utils.vertices.FrameBasedRecordingRegion(
            probe.size_in, assembler.n_ticks)

        return cls(system_region, input_filter_region, input_filter_routing,
                   recording_region, probe.probe)
"""
