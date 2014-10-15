"""Tools for building Nengo Ensemble objects into intermediate representations.
"""

import nengo
import numpy as np

from . import intermediate, lif
from . import connections as ensemble_connection_utils
from ..utils import connections as connection_utils


def build_ensembles(objects, connections, probes, dt, rng):
    """Build Ensembles and related connections into intermediate
    representation form.
    """
    # Sort out GlobalInhibitionConnections
    (objects, connections) = \
        ensemble_connection_utils.process_global_inhibition_connections(
            objects, connections, probes)

    # Separate out the set of Ensembles from the set of other objects
    ensembles, new_objects = split_out_ensembles(objects)

    # Replace Ensembles with an appropriate type of IntermediateEnsemble, add
    # these new objects to the list of final objects.
    replaced_ensembles = replace_ensembles(ensembles, dt, rng)
    new_objects.extend(replaced_ensembles.values())

    # Replace Connections which connected into or out of these objects with new
    # equivalents.
    new_connections = connection_utils.replace_objects_in_connections(
        connections, replaced_ensembles)

    # Modify IntermediateEnsembles to take account of any probes which are
    # recording them.
    apply_probing(replaced_ensembles, probes)

    # Add direct inputs, remove unnecessary connections
    remove_connections = include_constant_inputs(connections)
    new_connections = [c for c in connections if c not in remove_connections]

    return new_objects, new_connections


def split_out_ensembles(objects):
    """Splits a list of objects into a list of Ensembles and a list of others.
    """
    ensembles = [e for e in objects if isinstance(e, nengo.Ensemble)]
    others = [o for o in objects if not isinstance(o, nengo.Ensemble)]

    return ensembles, others


def replace_ensembles(ensembles, connections, dt, rng):
    """Replace Ensembles with intermediate representations of Ensembles.

    :returns dict: Map of ensembles to intermediate ensemble representation.
    """
    replaced_ensembles = dict()

    for ensemble in ensembles:
        # Build the appropriate intermediate representation for the Ensemble
        if isinstance(ensemble.neuron_type, nengo.neurons.LIF):
            # Get the set of outgoing connections for this Ensemble so that
            # decoders can be solved for.
            out_conns = [c for c in connections if c.pre_obj is ensemble]
            replaced_ensembles[ensemble] = \
                lif.IntermediateLIF.from_object(ensemble, out_conns, dt, rng)
        else:
            raise NotImplementedError(
                "nengo_spinnaker does not currently support '{}' neurons."
                .format(ensemble.neuron_type.__class__.__name__))

    return replaced_ensembles


def apply_probing(replaced_ensembles, probes):
    """Apply probes to the replaced Ensembles.
    """
    for p in probes:
        # Ignore probes for targets that are not ensembles
        if p.target not in replaced_ensembles:
            continue

        # Apply spike probing
        if p.attr == 'spikes':
            replaced_ensembles[p.target].record_spikes = True
        else:
            raise NotImplementedError(
                "Probing {} for {} objects is not currently supported.".
                format(p.attr, p.target.__class__.__name__)
            )
        replaced_ensembles[p.target].probes.append(p)


def include_constant_inputs(connections):
    """Remove connections from constant valued Nodes to IntermediateEnsembles.

    The value of the Node is included in the bias currents for the receiving
    Ensemble.  It assumed that (1) all ensembles have been built into
    IntermediateEnsemble objects and (2) that all pass nodes have already been
    removed.

    :return list: A list of connections to be removed from the network.
    """
    remove_connections = list()

    for c in connections:
        if (isinstance(c.post_obj, intermediate.IntermediateEnsemble) and
                isinstance(c.pre_obj, nengo.Node) and not
                callable(c.pre_obj.output)):
            # This Node just forms direct input, add it to direct input and
            # list the connection as requiring removal.
            inp = c.pre_obj.output
            if c.function is not None:
                inp = c.function(inp)
            c.post_obj.direct_input += np.dot(c.transform, inp)

            remove_connections.append(c)

    return remove_connections
