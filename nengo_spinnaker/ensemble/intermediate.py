"""Intermediate representations of Ensembles before they are made into
vertices.
"""
import collections

import numpy as np


IntermediateLearningRule = collections.namedtuple(
    'IntermediateLearningRule', 'rule decoder_index')


class IntermediateEnsemble(object):
    def __init__(self, n_neurons, gains, bias, encoders, decoders,
                 decoder_headers, learning_rules, direct_input, label=None):
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
        self.direct_input = direct_input
