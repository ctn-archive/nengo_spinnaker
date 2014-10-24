"""Utilities for building up a tree of connections from a set of connections.

A connection tree originates from a root node, the first layer of sub-trees
represent unique connection parameters with the leaves representing pairs of
filters and terminating objects.
Some operations are defined for these trees, such as replacing objects.
Generally, performing an operation on the tree results in the creation of a new
tree.  Only certain changes are allowed to alter the structure of the tree.
"""

import numpy as np


class ReducedConnection(object):
    __slots__ = ['transform', 'function', 'keyspace']

    # Comparisons between connections: ReducedConnections are equivalent iff.
    # they share a function, a keyspace, a transform and a class type.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.function is b.function,
        lambda a, b: np.all(a.transform == b.transform),
    ]

    def __init__(self, transform, function, keyspace=None):
        self.transform = np.array(transform).copy()
        self.transform.flags.writeable = False
        self.function = function
        self.keyspace = keyspace

    def copy_with_transform(self, transform):
        """Create a copy of this ReducedConnection but with the transform
        transformed by the given value or matrix.
        """
        return self.__class__(np.dot(transform, self.transform), self.function,
                              self.keyspace)

    def __repr__(self):
        return "<{:s} at {:#x}>".format(self.__class__.__name__, id(self))

    def __copy__(self):
        return self.__class__(self.transform, self.function, self.keyspace)

    def __hash__(self):
        return hash((self.__class__, self.transform.data, self.function,
                     self.keyspace))

    def __eq__(self, other):
        # Equality is managed by checking the result of a generator, _eqs
        return all(self.__eqs(other))

    def __eqs(self, other):
        for fn in self._eq_terms:
            yield fn(self, other)


class ReducedEnsembleConnection(ReducedConnection):
    __slots__ = ['eval_points', 'solver', 'transmitter_learning_rules']

    # ReducedEnsembleConnections are equivalent iff. they meet they share a
    # class, a keyspace, a solver, a transform, eval points and a function
    # (evaluated on those eval points).
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.solver == b.solver,
        lambda a, b: np.all(a.transform == b.transform),
        lambda a, b: np.all(a.eval_points == b.eval_points),
        lambda a, b: np.all(a._get_evaluated_function() ==
                            b._get_evaluated_function()),
    ]

    def __init__(self, transform, function, keyspace=None, eval_points=None,
                 solver=None):
        super(ReducedEnsembleConnection, self).__init__(transform, function,
                                                        keyspace)
        self.eval_points = np.array(eval_points).copy()
        self.eval_points.flags.writeable = False
        self.solver = solver

    def __hash__(self):
        return hash((self.__class__, self.transform.data, self.keyspace,
                     self.solver, self.eval_points.data,
                     self._get_evaluated_function().data))

    def _get_evaluated_function(self):
        """Evaluate the function at eval points and return Numpy array.
        """
        data = (self.function(self.eval_points) if self.function is not None
                else self.eval_points)
        data.flags.writeable = False
        return data
