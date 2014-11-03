import numpy as np


class PlaceholderEnsemble(object):
    __slots__ = ['ens', 'direct_input', 'record_spikes', 'probes',
                 '_eval_points']

    def __init__(self, ens, record_spikes=False):
        self.direct_input = np.zeros(ens.size_in)
        self.ens = ens
        self.record_spikes = record_spikes
        self.probes = list()
        self._eval_points = None

    @property
    def size_in(self):
        return self.ens.size_in

    @property
    def size_out(self):
        return self.ens.size_out

    @property
    def eval_points(self):
        if self._eval_points is None:
            return self.ens.eval_points
        return self._eval_points

    @eval_points.setter
    def eval_points(self, eval_points):
        self._eval_points = eval_points

    def __repr__(self):
        return "<PlaceholdEnsemble for {}>".format(self.ens)
