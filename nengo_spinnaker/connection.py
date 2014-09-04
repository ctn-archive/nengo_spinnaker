import nengo
import pacman103


class IntermediateConnection(object):
    """Intermediate representation of a connection object.
    """
    def __init__(self, pre_obj, post_obj, synapse=None, function=None,
                 transform=1., solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True, learning_rule=None, modulatory=False):
        self.pre_obj = pre_obj
        self.post_obj = post_obj
        self.synapse = synapse
        self.function = function
        self.transform = transform
        self.solver = solver
        self.eval_points = eval_points
        self.keyspace = keyspace
        self.width = post_obj.size_in
        self.is_accumulatory = is_accumulatory
        self.learning_rule = learning_rule
        self.modulatory = modulatory

        self.pre_slice = slice(None, None, None)
        self.post_slice = slice(None, None, None)

    def _required_transform_shape(self):
        return (self.pre_obj.size_out, self.post_obj.size_in)

    @classmethod
    def from_connection(cls, c):
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
            keyspace = getattr(c, 'keyspace', None)
            return cls(c.pre_obj, c.post_obj, synapse=c.synapse,
                       function=c.function, transform=tr, solver=c.solver,
                       eval_points=c.eval_points, keyspace=keyspace,
                       learning_rule=c.learning_rule, modulatory=c.modulatory)
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


class NengoEdge(pacman103.lib.graph.Edge):
    def __init__(self, prevertex, postvertex, keyspace):
        super(NengoEdge, self).__init__(prevertex, postvertex)
        self.keyspace = keyspace


def generic_connection_builder(connection, assembler):
    """Builder for connections which just require an edge between two vertices
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
