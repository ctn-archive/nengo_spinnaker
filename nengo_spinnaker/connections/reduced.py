import copy
import nengo
import numpy as np
import sentinel


class OutgoingReducedConnection(object):
    """Represents the limited information required to transmit data.

    The minimum set of parameters to transmit information are the transform
    provided on a connection, the function computed on the connection and the
    keyspace (if any) attached to the connection.
    """
    __slots__ = ['width', 'transform', 'function', 'pre_slice', 'post_slice',
                 'keyspace']

    # Comparisons between connections: ReducedConnections are equivalent iff.
    # they share a function, a keyspace, a transform and a class type.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.pre_slice == b.pre_slice,
        lambda a, b: a.post_slice == b.post_slice,
        lambda a, b: a.width == b.width,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.function is b.function,
        lambda a, b: np.all(a.transform == b.transform),
    ]

    def __init__(self, width, transform, function, pre_slice, post_slice,
                 keyspace=None):
        self.width = width
        self.transform = np.array(transform).copy()
        self.transform.flags.writeable = False
        self.function = function
        self.pre_slice = pre_slice
        self.post_slice = post_slice
        self.keyspace = keyspace

    def __repr__(self):  # pragma: no cover
        return "<{:s} at {:#x}>".format(self.__class__.__name__, id(self))

    def __str__(self):  # pragma: no cover
        return "{}({})<{}>".format(self.__class__.__name__, self.keyspace,
                                   self.width)

    def __copy__(self):
        return self.__class__(
            self.width, self.transform, self.function,
            copy.copy(self.pre_slice), copy.copy(self.post_slice),
            self.keyspace)

    def __hash__(self):
        return hash((self.__class__, self.width, self.transform.data,
                     self.function, self.keyspace, hash_slice(self.pre_slice),
                     hash_slice(self.post_slice)))

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
    __slots__ = ['eval_points', 'solver', 'transmitter_learning_rule']

    # ReducedEnsembleConnections are equivalent iff. they meet they share a
    # class, a keyspace, a solver, a transform, eval points, a function
    # (evaluated on those eval points) and have NO learning rules.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.width == b.width,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.solver == b.solver,
        lambda a, b: a.transmitter_learning_rule is None,
        lambda a, b: b.transmitter_learning_rule is None,
        lambda a, b: np.all(a.transform == b.transform),
        lambda a, b: np.all(a.eval_points == b.eval_points),
        lambda a, b: np.all(a._get_evaluated_function() ==
                            b._get_evaluated_function()),
    ]

    def __init__(self, width, transform, function, pre_slice, post_slice,
                 keyspace=None, eval_points=None, solver=None,
                 transmitter_learning_rule=None):
        super(OutgoingReducedEnsembleConnection, self).__init__(
            width, transform, function, pre_slice, post_slice, keyspace)
        self.eval_points = np.array(eval_points).copy()
        self.eval_points.flags.writeable = False
        self.solver = solver
        self.transmitter_learning_rule = transmitter_learning_rule

    def __copy__(self):
        return self.__class__(
            self.width, self.transform, self.function,
            copy.copy(self.pre_slice), copy.copy(self.post_slice),
            self.keyspace, self.eval_points, self.solver,
            self.transmitter_learning_rule)

    def __hash__(self):
        return hash((self.__class__, self.transform.data, self.keyspace,
                     self.solver, self.eval_points.data,
                     hash_slice(self.pre_slice), hash_slice(self.post_slice),
                     self._get_evaluated_function().data,
                     self.transmitter_learning_rule))

    def _get_evaluated_function(self):
        """Evaluate the function at eval points and return Numpy array.
        """
        data = (self.function(self.eval_points) if self.function is not None
                else self.eval_points)
        data.flags.writeable = False
        return data


# These are port objects that are valid on all appropriate objects, various
# optimisations may alter the ports that are addressed by connections.  For
# ensembles additional ports may be required (for example, receiving error)
# signals, these are represented with a reference to the outgoing connection
# with a learning rule.
StandardInputPort = sentinel.create('StandardInput')
GlobalInhibitionPort = sentinel.create('GlobalInhibitionPort')


class Target(object):
    """Represents the convergence of a signal on a specific port of an object.
    """
    __slots__ = ['target_object', 'slice', 'port']

    def __init__(self, target_object, slice, port=StandardInputPort):
        self.target_object = target_object
        self.port = port
        self.slice = slice

    def __eq__(self, other):
        # Targets are equivalent iff. they refer to the same port on the same
        # object.  Some standard ports will be defined.
        return all([
            self.target_object is other.target_object,
            self.slice == other.slice,
            self.port is other.port,
        ])

    def __copy__(self):
        return self.__class__(self.target_object, copy.copy(self.slice),
                              self.port)

    def __hash__(self):
        return hash((self.target_object, hash_slice(self.slice), self.port))

    def __str__(self):  # pragma: no cover
        return "{}.{}".format(self.target_object, self.port)


# Filter parameters represent the filtering that is applied to values received
# by Nengo executables.
class FilterParameter(object):
    """Base class for filter types."""
    __slots__ = ['is_accumulatory']

    def __init__(self, is_accumulatory=True):
        self.is_accumulatory = is_accumulatory

    def __eq__(self, other):
        return all([
            self.__class__ is other.__class__,
            self.is_accumulatory == other.is_accumulatory,
        ])

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((hash(self.__class__), hash(self.is_accumulatory)))


class LowpassFilterParameter(FilterParameter):
    __slots__ = ['tau']

    def __init__(self, tau, is_accumulatory=True):
        super(LowpassFilterParameter, self).__init__(is_accumulatory)
        self.tau = tau if tau is not None else 0.

    @classmethod
    def from_synapse(cls, synapse, is_accumulatory=True):
        return cls(synapse.tau, is_accumulatory)

    def __eq__(self, other):
        return all([
            super(LowpassFilterParameter, self).__eq__(other),
            self.tau == other.tau
        ])

    def __hash__(self):
        return hash((super(LowpassFilterParameter, self).__hash__(),
                     hash(self.tau)))

    def __repr__(self):  # pragma: no cover
        return "LowpassFilterParameter({:.3f}, is_accumulatory={})".format(
            self.tau, self.is_accumulatory)


_filter_types = {nengo.Lowpass: LowpassFilterParameter, }  # pragma: no cover


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
        lambda a, b: a.target == b.target,
        lambda a, b: a.filter_object == b.filter_object,
    ]

    def __init__(self, target, filter_object):
        self.target = target
        self.filter_object = filter_object

    def __eq__(self, other):
        return all(fn(self, other) for fn in self._eq_terms)

    def __hash__(self):
        return hash((self.target, self.filter_object))

    def __copy__(self):
        return self.__class__(copy.copy(self.target),
                              copy.copy(self.filter_object))

    def __repr__(self):  # pragma: no cover
        return "<{} {:s}>".format(self.__class__.__name__, self.target)

    def __str__(self):  # pragma: no cover
        return "{} terminating at {}".format(self.__class__.__name__,
                                             self.target)


def hash_slice(slice):
    return hash((slice.start, slice.stop, slice.step))
