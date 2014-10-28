"""Connections that can be mapped onto SpiNNaker communication channels.

Connection Trees
----------------
A connection tree originates from a root node, the first layer of sub-trees
represent unique connection parameters with the leaves representing pairs of
filters and terminating objects.
Some operations are defined for these trees, such as replacing objects.
Generally, performing an operation on the tree results in the creation of a new
tree.  Only certain changes are allowed to alter the structure of the tree.
"""

import collections
import enum
import nengo
import numpy as np

from pacman.model.partitionable_graph.partitionable_edge import (
    PartitionableEdge as PacmanPartitionableEdge)


FunctionInfo = collections.namedtuple('FunctionInfo', 'function size')


class IntermediateConnection(object):
    """Intermediate representation of a connection object.
    """
    def __init__(self, pre_obj, post_obj, synapse=None, function=None,
                 transform=1., solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True, learning_rule=None, modulatory=False):
        # Handle sliced connections correctly
        if isinstance(pre_obj, nengo.base.ObjView):
            self.pre_obj = pre_obj.obj
            self.pre_slice = pre_obj.slice
        else:
            self.pre_obj = pre_obj
            self.pre_slice = slice(None, None, None)

        if isinstance(post_obj, nengo.base.ObjView):
            self.post_obj = post_obj.obj
            self.post_slice = post_obj.slice
        else:
            self.post_obj = post_obj
            self.post_slice = slice(None, None, None)

        self.transform = np.array(transform)

        self.synapse = synapse
        self.function = function
        self.solver = solver
        self.eval_points = eval_points
        self.keyspace = keyspace
        self.width = post_obj.size_in
        self.is_accumulatory = is_accumulatory
        self.learning_rule = learning_rule
        self.modulatory = modulatory

        if any(sl != slice(None) for sl in [self.pre_slice, self.post_slice]):
            # ***YUCK***
            # Determine the size requirements of the function
            if self.function is not None:
                raise NotImplementedError
            else:
                self.function_info = FunctionInfo(None, None)

            self.transform = nengo.utils.builder.full_transform(
                self, allow_scalars=False)

    def _required_transform_shape(self):
        # TODO This is no longer required?
        return (self.pre_obj.size_out, self.post_obj.size_in)

    @classmethod
    def from_connection(cls, c, keyspace=None, is_accumulatory=True):
        """Return an IntermediateConnection object for any connections which
        have not already been replaced.  A requirement of any replaced
        connection type is that it has the attribute keyspace and can have
        its pre_obj and post_obj amended by later functions.
        """
        if isinstance(c, nengo.Connection):
            # Get the full transform
            tr = nengo.utils.builder.full_transform(c, allow_scalars=False)

            # Return a copy of this connection but with less information and
            # the full transform.
            return cls(c.pre_obj, c.post_obj, synapse=c.synapse,
                       function=c.function, transform=tr, solver=c.solver,
                       eval_points=c.eval_points, keyspace=keyspace,
                       learning_rule=c.learning_rule, modulatory=c.modulatory,
                       is_accumulatory=is_accumulatory)
        return c

    def to_connection(self):
        """Create a standard Nengo connection from this object.
        """
        return nengo.Connection(self.pre_obj, self.post_obj,
                                synapse=self.synapse, function=self.function,
                                transform=self.transform, solver=self.solver,
                                eval_points=self.eval_points,
                                learning_rule=self.learning_rule,
                                modulatory=self.modulatory,
                                add_to_container=False)

    def __repr__(self):
        return '<IntermediateConnection({}, {})>'.format(self.pre_obj,
                                                         self.post_obj)

    def get_reduced_outgoing_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        # Generate the outgoing component, depending on the nature of the
        # originating object.
        if not isinstance(self.pre_obj, nengo.Ensemble):
            return OutgoingReducedConnection(
                self.transform, self.function, self.keyspace)
        else:
            if self.learning_rule is not None:
                # Check the learning rule affects transmission and include it.
                raise NotImplementedError
            return OutgoingReducedEnsembleConnection(
                self.transform, self.function, self.keyspace, self.eval_points,
                self.solver)

    def get_reduced_incoming_connection(self):
        """Convert the IntermediateConnection into a reduced representation.
        """
        return IncomingReducedConnection(Target(self.post_obj),
                                         self._get_filter())

    def _get_filter(self):
        try:
            return _filter_types[self.synapse.__class__].from_synapse(
                self.width, self.synapse, self.is_accumulatory)
        except KeyError:
            raise NotImplementedError(self.synapse.__class__)


class NengoEdge(PacmanPartitionableEdge):
    def __init__(self, prevertex, postvertex, keyspace):
        super(NengoEdge, self).__init__(prevertex, postvertex)
        self.keyspace = keyspace


def generic_connection_builder(connection, assembler):
    """Builder for connections which just require an edge between two vertices.
    """
    # Get the pre_obj and post_obj objects
    prevertex = assembler.get_object_vertex(connection.pre_obj)
    postvertex = assembler.get_object_vertex(connection.post_obj)

    if prevertex is None or postvertex is None:
        return

    # Ensure that the keyspace is set
    assert connection.keyspace is not None

    # Create and return the edge
    return NengoEdge(prevertex, postvertex, connection.keyspace)


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
    __slots__ = ['eval_points', 'solver', 'transmitter_learning_rule']

    # ReducedEnsembleConnections are equivalent iff. they meet they share a
    # class, a keyspace, a solver, a transform, eval points, a function
    # (evaluated on those eval points) and have NO learning rules.
    _eq_terms = [
        lambda a, b: a.__class__ is b.__class__,
        lambda a, b: a.keyspace == b.keyspace,
        lambda a, b: a.solver == b.solver,
        lambda a, b: a.transmitter_learning_rule is None,
        lambda a, b: b.transmitter_learning_rule is None,
        lambda a, b: np.all(a.transform == b.transform),
        lambda a, b: np.all(a.eval_points == b.eval_points),
        lambda a, b: np.all(a._get_evaluated_function() ==
                            b._get_evaluated_function()),
    ]

    def __init__(self, transform, function, keyspace=None, eval_points=None,
                 solver=None, transmitter_learning_rule=None):
        super(OutgoingReducedEnsembleConnection, self).__init__(
            transform, function, keyspace)
        self.eval_points = np.array(eval_points).copy()
        self.eval_points.flags.writeable = False
        self.solver = solver
        self.transmitter_learning_rule = transmitter_learning_rule

    def __hash__(self):
        return hash((self.__class__, self.transform.data, self.keyspace,
                     self.solver, self.eval_points.data,
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
StandardPorts = enum.Enum('StandardPorts', 'INPUT')
EnsemblePorts = enum.Enum('EnsemblePorts', 'GLOBAL_INHIBITION')


class Target(object):
    """Represents the convergence of a signal on a specific port of an object.
    """
    __slots__ = ['target_object', 'port']

    def __init__(self, target_object, port=StandardPorts.INPUT):
        self.target_object = target_object
        self.port = port

    def __eq__(self, other):
        # Targets are equivalent iff. they refer to the same port on the same
        # object.  Some standard ports will be defined.
        return all([
            self.target_object is other.target_object,
            self.port is other.port,
        ])

    def __hash__(self):
        return hash((self.target_object, self.port))

    def __str__(self):
        return "{}.{}".format(self.target_object, self.port)


class FilterParameter(object):
    """Base class for filter types."""
    def __init__(self, width, is_accumulatory=True, is_modulatory=False):
        # TODO: Is width actually required?  Keep it to maintain compatibility
        #       current C code, but investigate removing it if we can.  Ideally
        #       width is inherently part of the port filter sits on.
        self.width = width
        self.is_accumulatory = is_accumulatory
        self.is_modulatory = is_modulatory

    @classmethod
    def from_synapse(cls, synapse, is_accumulatory=True, is_modulatory=False):
        raise NotImplementedError

    def __eq__(self, other):
        return all([
            self.__class__ is other.__class__,
            self.is_accumulatory == other.is_accumulatory,
            self.is_modulatory == other.is_modulatory,
            self.width == other.width,
        ])

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((hash(self.__class__), hash(self.is_accumulatory),
                     hash(self.is_modulatory), hash(self.width)))


class LowpassFilterParameter(FilterParameter):
    def __init__(self, width, tau, is_accumulatory=True, is_modulatory=False):
        super(LowpassFilterParameter, self).__init__(width, is_accumulatory,
                                                     is_modulatory)
        self.tau = tau if tau is not None else 0.

    @classmethod
    def from_synapse(cls, width, synapse, is_accumulatory=True,
                     is_modulatory=False):
        return cls(width, synapse.tau, is_accumulatory, is_modulatory)

    def __eq__(self, other):
        return all([
            super(LowpassFilterParameter, self).__eq__(other),
            self.tau == other.tau
        ])

    def __hash__(self):
        return hash((super(LowpassFilterParameter, self).__hash__(),
                     hash(self.tau)))

    def __repr__(self):
        return "LowpassFilterParameter({:d}, {:.3f}, {}, {})".format(
            self.width, self.tau, self.is_accumulatory, self.is_modulatory)


_filter_types = {nengo.Lowpass: LowpassFilterParameter, }


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

    def __repr__(self):
        return "<{} {:s}>".format(self.__class__.__name__, self.target)


def build_connection_trees(connections):
    """Constructs the connection trees for the model.

    The connection "tree" consists of a set of root nodes, each of which
    represents an object which transmits data to other objects.  The first set
    of branches represents outgoing connections from each object, these will be
    automatically merged and reduced.  The final lists of leaves represent each
    object, port and filter which receives the data transmitted over each
    connection.
    """
    # For example, the model:
    #
    #     (a) --CONN1-> (b)
    #      | \           |
    #      |  \-CONN1----/
    #      |
    #      \--CONN2---> (c)
    #
    # Will result in a "forest":
    #
    #     a -----> (CONN1) -----> b.INPUT
    #       \              \----> b.INPUT
    #        \---> (CONN2) -----> c.INPUT
    #
    # Meaning that each packet `a' transmits over CONN1 will be "received
    # twice" by `b', packets transmitted by `a' over CONN2 will only be
    # received by `c'.

    # Create the empty tree
    # The leaves of this tree are NOT sets because packets may be processed
    # twice by the same filter.
    tree = collections.defaultdict(lambda: collections.defaultdict(list))

    # For each connection generate the outgoing and incoming connections and
    # add to the dictionary.
    for c in connections:
        # Get the reduced outgoing and incoming connection parts
        outgoing = c.get_reduced_outgoing_connection()
        incoming = c.get_reduced_incoming_connection()

        # Add these connections to the tree
        tree[c.pre_obj][outgoing].append(incoming)

    return tree


def get_incoming_connections_from_tree(conn_tree, obj):
    """Retrieve the reduced incoming connections for a given object.

    A dictionary of incoming connections by port will be generated, each entry
    will be a map of filters to a list of outgoing connections which feed it.
    """
    # This is implemented as a leaf walk on the connection tree which adds each
    # leaf which represents a connection terminating at `obj' to the dict of
    # port filters.
    # The leaves of this new tree are NOT sets because packets may be processed
    # twice by the same filter.
    tree = collections.defaultdict(lambda: collections.defaultdict(list))

    # Iterate over all originating nodes,
    for (_, outgoing_conns) in conn_tree.iteritems():
        # and all outgoing connections,
        for (outgoing, ins) in outgoing_conns.iteritems():
            # and all incoming connections.
            for incoming in ins:
                # If this incoming connection refers to the target object then
                # there is a connection arriving at this object.
                target_obj = incoming.target.target_object
                port = incoming.target.port

                if target_obj is obj:
                    # Store a reference to the outgoing connection which
                    # terminates at this port and filter.
                    tree[port][incoming.filter_object].append(outgoing)
    return tree


def get_objects_from_connection_trees(connection_tree):
    """Return a list of objects which are contained within the connection tree.
    """
    objects = set()
    
    # For each originating object and set of outgoing connections
    for (o, conns) in connection_tree.items():
        # Add the object
        objects.add(o)

        # For each set of incoming connections branching of an outgoing
        # connection.
        for iconns in conns.values():
            # For each incoming connection
            for c in iconns:
                # Add the target object of that connection
                objects.add(c.target.target_object)

    return list(objects)
