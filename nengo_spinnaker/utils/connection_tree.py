"""Utilities for building up a tree of connections from a set of connections.

A connection tree originates from a root node, the first layer of sub-trees
represent unique connection parameters with the leaves representing pairs of
filters and terminating objects.
Some operations are defined for these trees, such as replacing objects.
Generally, performing an operation on the tree results in the creation of a new
tree.  Only certain changes are allowed to alter the structure of the tree.
"""

import numpy as np


class OutgoingReducedConnection(object):
    """Represents the limited information required to transmit data.

    The minimum set of parameters to transmit information are the transform
    provided on a connection, the function computed on the connection and the
    keyspace (if any) attached to the connection.
    """
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
        return all(fn(self, other) for fn in self._eq_terms)


class OutgoingReducedEnsembleConnection(OutgoingReducedConnection):
    """Represents the limited information required to transmit ensemble data.

    The minimum set of parameters to transmit information are the transform
    provided on a connection, the function computed on the connection, the
    keyspace (if any) attached to the connection; for ensembles some additional
    components are necessary: the evaluation points for decoder solving, the
    specific solver and any learning rules which modify the transmitted value.
    """
    __slots__ = ['eval_points', 'solver', 'transmitter_learning_rules']

    # ReducedEnsembleConnections are equivalent iff. they meet they share a
    # class, a keyspace, a solver, a transform, eval points, a function
    # (evaluated on those eval points) and have NO learning rules.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.solver == b.solver,
        lambda a, b: len(a.transmitter_learning_rules) == 0,
        lambda a, b: len(b.transmitter_learning_rules) == 0,
        lambda a, b: np.all(a.transform == b.transform),
        lambda a, b: np.all(a.eval_points == b.eval_points),
        lambda a, b: np.all(a._get_evaluated_function() ==
                            b._get_evaluated_function()),
    ]

    def __init__(self, transform, function, keyspace=None, eval_points=None,
                 solver=None, transmitter_learning_rules=list()):
        super(OutgoingReducedEnsembleConnection, self).__init__(
            transform, function, keyspace)
        self.eval_points = np.array(eval_points).copy()
        self.eval_points.flags.writeable = False
        self.solver = solver
        self.transmitter_learning_rules = tuple(transmitter_learning_rules)

    def __hash__(self):
        return hash((self.__class__, self.transform.data, self.keyspace,
                     self.solver, self.eval_points.data,
                     self._get_evaluated_function().data,
                     self.transmitter_learning_rules))

    def _get_evaluated_function(self):
        """Evaluate the function at eval points and return Numpy array.
        """
        data = (self.function(self.eval_points) if self.function is not None
                else self.eval_points)
        data.flags.writeable = False
        return data


class IncomingReducedConnection(object):
    """Represents the limited information required to receive data.

    The minimum set of parameters to transmit information are the object that
    is receiving the data and the filter used.
    """
    __slots__ = ['target', 'filter_object']

    # Incoming reduced connections are equivalent iff. they share a receiving
    # object (target) and have equivalent connections.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.target is b.target,
        lambda a, b: a.filter_object == b.filter_object,
    ]

    def __init__(self, target, filter_object):
        self.target = target
        self.filter_object = filter_object

    def __eq__(self, other):
        return all(fn(self, other) for fn in self._eq_terms)

    def __hash__(self):
        return hash((self.target, self.filter_object))


class IncomingReducedEnsembleConnection(IncomingReducedConnection):
    """Represents the limited information required to receive data.

    The minimum set of parameters to transmit information are the object that
    is receiving the data and the filter used.
    """
    __slots__ = ['receiver_learning_rules']

    # Incoming reduced connections are equivalent iff. they share a receiving
    # object (target), have equivalent connections and have NO learning rules.
    _eq_terms = IncomingReducedConnection._eq_terms + [
        lambda a, b: len(a.receiver_learning_rules) == 0,
        lambda a, b: len(b.receiver_learning_rules) == 0,
    ]

    def __init__(self, target, filter_object, receiver_learning_rules=list()):
        super(IncomingReducedEnsembleConnection, self).__init__(
            target, filter_object)
        self.receiver_learning_rules = tuple(receiver_learning_rules)

    def __hash__(self):
        return hash((super(IncomingReducedEnsembleConnection, self).__hash__(),
                     self.receiver_learning_rules))
