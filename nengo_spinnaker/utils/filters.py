"""Utilities for representing synapses/filters.
"""

import collections
import nengo


class SynapseParameter(object):
    def __init__(self, is_accumulatory=False, is_modulatory=False):
        self.is_accumulatory = is_accumulatory
        self.is_modulatory = is_modulatory

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.is_accumulatory == other.is_accumulatory and
                self.is_modulatory == other.is_modulatory)

    def __hash__(self):
        return hash((hash(self.__class__), hash(self.is_accumulatory),
                     hash(self.is_modulatory)))


class LinearFilterSynapseParameter(SynapseParameter):
    def __init__(self, synapse, is_accumulatory=False, is_modulatory=False):
        super(LinearFilterSynapseParameter, self).__init__(is_accumulatory,
                                                           is_modulatory)
        self.num = synapse.num
        self.den = synapse.den

    def __eq__(self, other):
        return (super(LinearFilterSynapseParameter, self).__eq__(other) and
                self.num == other.num and
                self.den == other.den)

    def __hash__(self):
        return hash((super(LinearFilterSynapseParameter, self).__hash__(),
                     hash(self.num), hash(self.den)))


class LowpassSynapseParameter(SynapseParameter):
    def __init__(self, synapse, is_accumulatory=False, is_modulatory=False):
        super(LowpassSynapseParameter, self).__init__(is_accumulatory,
                                                      is_modulatory)
        self.tau = synapse.tau

    def __eq__(self, other):
        return (super(LowpassSynapseParameter, self).__eq__(other) and
                self.tau == other.tau)

    def __hash__(self):
        return hash((super(LowpassSynapseParameter, self).__hash__(),
                     hash(self.tau)))


class AlphaSynapseParameter(LowpassSynapseParameter):
    # Obviously this is a different type, but it has the same parameters as the
    # lowpass and so can derive from it safely (DRY).
    pass


_FilterTypes = {nengo.synapses.LinearFilter: LinearFilterSynapseParameter,
                nengo.Lowpass: LowpassSynapseParameter,
                nengo.synapses.Alpha: AlphaSynapseParameter, }


def get_combined_filters(connections):
    """Return the minimum set of filters required to filter given connections.

    Also returns a mapping of connection to filter index.
    """
    # Create a dictionary of filter to connections
    filter_connections = collections.defaultdict(list)
    for c in connections:
        f = _FilterTypes[c.synapse.__class__](
            c.synapse, getattr(c, 'is_accumulatory', False), c.modulatory)
        filter_connections[f].append(c)

    # Create a list of filters and a mapping of connection to filter index.
    filters = list()
    filter_indices = dict()

    for (f, cs) in filter_connections.iteritems():
        filters.append(f)

        for c in cs:
            filter_indices[c] = filters.index(f)

    return filters, filter_indices
