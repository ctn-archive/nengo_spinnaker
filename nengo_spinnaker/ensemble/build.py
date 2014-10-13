"""Tools for building Nengo Ensemble objects into intermediate representations.
"""

import nengo
import numpy as np

from . import (intermediate, lif)
from . import connections as ensemble_connection_utils


def build_ensembles(objects, connections, probes, dt, rng):
    """Build Ensembles and related connections into intermediate
    representation form.
    """
    new_objects = list()
    new_connections = list()

    # Sort out GlobalInhibitionConnections
    (objects, connections) = \
        ensemble_connection_utils.process_global_inhibition_connections(
            objects, connections, probes)

    # Create an intermediate representation for each Ensemble
    for obj in objects:
        if not isinstance(obj, nengo.Ensemble):
            new_objects.append(obj)
            continue

        # Build the appropriate intermediate representation for the Ensemble
        if isinstance(obj.neuron_type, nengo.neurons.LIF):
            # Get the set of outgoing connections for this Ensemble so that
            # decoders can be solved for.
            out_conns = [c for c in connections if c.pre_obj == obj]
            new_obj = lif.IntermediateLIF.from_object(obj, out_conns, dt, rng)
            new_objects.append(new_obj)
        else:
            raise NotImplementedError(
                "nengo_spinnaker does not currently support '{}' neurons."
                .format(obj.neuron_type.__class__.__name__))

        # Modify connections into/out of this ensemble
        for c in connections:
            if c.pre_obj is obj:
                c.pre_obj = new_obj
            if c.post_obj is obj:
                c.post_obj = new_obj

        # Mark the Ensemble as recording spikes/voltages if appropriate
        for p in probes:
            if p.target is obj:
                if p.attr == 'spikes':
                    new_obj.record_spikes = True
                    new_obj.probes.append(p)
                elif p.attr == 'voltage':
                    raise NotImplementedError("Voltage probing not currently "
                                              "supported.")
                    new_obj.record_voltage = True
                    new_obj.probes.append(p)

    # Add direct inputs
    for c in connections:
        if (isinstance(c.post_obj, intermediate.IntermediateEnsemble) and
                isinstance(c.pre_obj, nengo.Node) and not
                callable(c.pre_obj.output)):
            # This Node just forms direct input, add it to direct input and
            # don't add the connection to the list of new connections
            inp = c.pre_obj.output
            if c.function is not None:
                inp = c.function(inp)
            c.post_obj.direct_input += np.dot(c.transform, inp)
        else:
            new_connections.append(c)

    return new_objects, new_connections
