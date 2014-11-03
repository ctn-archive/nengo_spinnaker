"""Tools for building Nengo Ensemble objects into intermediate representations.
"""

import nengo
import numpy as np
import warnings

import nengo.utils.builder as builder_utils
from nengo.utils import distributions as dists

from . import lif, pes
from .placeholder import PlaceholderEnsemble
from . import connections as ensemble_connection_utils
from ..builder import Builder
from ..utils import connections as connection_utils


@Builder.network_transform
def build_ensembles(objects, connections, probes, rngs):
    """Build Ensembles into a very reduced form which can be further built
    later.
    """
    # Sort out GlobalInhibitionConnections
    (objects, connections) = \
        ensemble_connection_utils.process_global_inhibition_connections(
            objects, connections, probes)

    # Process PES connections
    (objects, connections) = pes.process_pes_connections(objects, connections,
                                                         probes)

    # Separate out the set of Ensembles from the set of other objects
    ensembles, new_objects = split_out_ensembles(objects)

    # Replace Ensembles with a placeholder that can be elaborated upon later.
    replaced_ensembles = create_placeholder_ensembles(ensembles)

    # Generate evaluation points for Ensembles
    for ens in ensembles:
        if isinstance(ens.eval_points, dists.Distribution):
            n_points = ens.n_eval_points
            if n_points is None:
                n_points = builder_utils.default_n_eval_points(
                    ens.n_neurons, ens.dimensions)
            eval_points = ens.eval_points.sample(
                n_points, ens.dimensions, rngs[ens])
            eval_points *= ens.radius
        else:
            if all([ens.eval_points is not None,
                    ens.eval_points.shape[0] != ens.n_eval_points]):
                warnings.warn(
                    "Number of eval points doesn't match n_eval_points. "
                    "Ignoring n_eval_points."
                )
            eval_points = np.array(ens.eval_points, dtype=np.float64)
        replaced_ensembles[ens].eval_points = eval_points

    new_objects.extend(replaced_ensembles.values())

    # Replace Connections which connected into or out of these objects with new
    # equivalents.
    new_connections = connection_utils.replace_objects_in_connections(
        connections, replaced_ensembles)

    # Modify IntermediateEnsembles to take account of any probes which are
    # recording them.
    apply_probing(replaced_ensembles, probes)

    # Add direct inputs, remove unnecessary connections
    remove_connections = include_constant_inputs(new_connections)
    for r in remove_connections:
        new_connections.remove(r)

    return new_objects, new_connections


ensemble_build_fns = {
    nengo.neurons.LIF: lif.IntermediateLIF.build,
}


@Builder.object_builder(PlaceholderEnsemble)
def build_ensemble(placeholder, connection_trees, config, rngs):
    """Build a single placeholder into an Intermediate Ensemble.
    """
    if placeholder.ens.neuron_type.__class__ not in ensemble_build_fns:
        # Neuron type not supported
        raise NotImplementedError(
            "nengo_spinnaker does not currently support neurons of type"
            " '{}'".format(placeholder.ens.neuron_type.__class__.__name__))

    return ensemble_build_fns[placeholder.ens.neuron_type.__class__](
        placeholder, connection_trees, config, rngs[placeholder.ens])


def split_out_ensembles(objects):
    """Splits a list of objects into a list of Ensembles and a list of others.
    """
    ensembles = [e for e in objects if isinstance(e, nengo.Ensemble)]
    others = [o for o in objects if not isinstance(o, nengo.Ensemble)]

    return ensembles, others


def create_placeholder_ensembles(ensembles):
    """Replace Ensembles with placeholders that can be further refined later.
    """
    replaced_ensembles = dict()

    for ens in ensembles:
        replaced_ensembles[ens] = PlaceholderEnsemble(ens, record_spikes=False)

    return replaced_ensembles


def apply_probing(replaced_ensembles, probes):
    """Apply probes to the replaced Ensembles.
    """
    # Build a map of neurons to placeholder
    neurons = {k.neurons: v for (k, v) in replaced_ensembles.items()}

    for p in probes:
        if p.target in neurons:
            # Spike probing
            if p.attr == 'output':
                neurons[p.target].record_spikes = True
            else:
                raise NotImplementedError(
                    "Probing {} for {} objects is not currently supported.".
                    format(p.attr, p.target.__class__.__name__)
                )
            neurons[p.target].probes.append(p)
        elif p.target in replaced_ensembles:
            # Voltage / other probing
            raise NotImplementedError(
                "Probing {} for {} objects is not currently supported.".
                format(p.attr, p.target.__class__.__name__)
            )


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
        if (isinstance(c.post_obj, PlaceholderEnsemble) and
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
