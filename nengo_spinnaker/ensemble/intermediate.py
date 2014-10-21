"""Intermediate representations of Ensembles before they are made into
vertices.
"""
import collections

import numpy as np


IntermediateLearningRule = collections.namedtuple(
    'IntermediateLearningRule', 'rule decoder_index')


class IntermediateEnsemble(object):
    def __init__(self, n_neurons, gains, bias, encoders, decoders,
                 eval_points, decoder_headers, learning_rules, label=None):
        self.n_neurons = n_neurons
        self.label = label

        # Assert that the number of neurons is reflected in other parameters
        assert gains.size == n_neurons
        assert bias.size == n_neurons
        assert encoders.shape[0] == n_neurons

        # Get the number of dimensions represented and store fundamental
        # parameters
        self.size_in = self.n_dimensions = encoders.shape[1]
        self.gains = gains
        self.bias = bias
        self.encoders = encoders
        self.decoders = decoders

        # Output keys
        self.decoder_headers = decoder_headers
        self.output_keys = list()

        # Learning rules
        self.learning_rules = learning_rules

        # Recording parameters
        self.record_spikes = False
        self.record_voltage = False
        self.probes = list()

        # Direct input
        self.direct_input = np.zeros(self.n_dimensions)

    def create_output_keyspaces(self, ens_id, keyspace):
        self.output_keyspaces = \
            _create_output_keyspaces(self.decoder_headers, ens_id, keyspace)


def _create_output_keyspaces(decoder_headers, ens_id, keyspace):
    """Constructs a list of output keyspaces.

    As decoders may have been compressed a set of output keys needs to be
    constructed for the columns of the combined decoders that exist.  A
    default keyspace is used if one is not provided within the decoder
    decoder headers.  The decoder headers take the form of a 3-tuple
    (keyspace, connection index, dimension index).

    :param int ens_id: The unique object ID given to the Ensemble.
    :param keyspace: A default keyspace to use to construct the keys.
    """
    output_keyspaces = list()
    for header in decoder_headers:
        ks = keyspace if header[0] is None else header[0]
        ks = ks(o=ens_id, i=header[1], d=header[2])
        output_keyspaces.append(ks)
    return output_keyspaces
