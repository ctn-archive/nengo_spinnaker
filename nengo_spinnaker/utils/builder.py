"""Some builder delegates.
"""
import collections

import nengo
from nengo.utils.builder import objs_and_connections, remove_passthrough_nodes
from . import connections, probes, nodes


PreparedNetwork = collections.namedtuple(
    'PreparedNetwork', ['objects', 'connections', 'probes', 'host_model'])


def prepare_network(model, io, config):
        """Takes a model, returns a flattened version with objects and
        connections modified as required for building for SpiNNaker.  Also
        returns a version of the network modified for simulation on the
        host.
        """
        # Flatten the network
        (objs, conns) = objs_and_connections(model)

        # Remove synapses from Probes(PassNode) where the PassNode has incoming
        # synapses -- provides a RuntimeWarning that this is happening
        prbs = probes.get_corrected_probes(model.probes, conns)

        # Add new Nodes to represent decoded_value probes
        (n_objs, n_conns) = probes.get_probe_nodes_connections(prbs)
        objs.extend(n_objs)
        conns.extend(n_conns)

        # Remove all the PassNodes
        (objs, conns) = remove_passthrough_nodes(
            objs, conns)

        # Generate a version of the network to simulate on the host
        host_network = nodes.create_host_network(
            [n for n in objs if isinstance(n, nengo.Node)], conns, io, config)

        return PreparedNetwork(objs, conns, prbs, host_network)
