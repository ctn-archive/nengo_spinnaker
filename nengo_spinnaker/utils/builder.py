import collections
import numpy as np

import nengo
from nengo.utils.builder import full_transform


class IntermediateConnection(object):
    """Intermediate representation of a connection object.
    """
    def __init__(self, pre, post, synapse=None, function=None, transform=1.,
                 solver=None, eval_points=None, keyspace=None,
                 is_accumulatory=True):
        self.pre = pre
        self.post = post
        self.synapse = synapse
        self.function = function
        self.transform = transform
        self.solver = solver
        self.eval_points = eval_points
        self.keyspace = keyspace
        self.width = post.size_in
        self.is_accumulatory = is_accumulatory

        self._preslice = slice(None, None, None)
        self._postslice = slice(None, None, None)

    def _required_transform_shape(self):
        return (self.pre.size_out, self.post.size_in)

    @classmethod
    def from_connection(cls, c):
        """Return an IntermediateConnection object for any connections which
        have not already been replaced.  A requirement of any replaced
        connection type is that it has the attribute keyspace and can have
        its pre and post amended by later functions.
        """
        if isinstance(c, nengo.Connection):
            # Get the full transform
            tr = nengo.utils.builder.full_transform(c, allow_scalars=False)

            # Return a copy of this connection but with less information and
            # the full transform.
            keyspace = getattr(c, 'keyspace', None)
            return cls(c.pre, c.post, c.synapse, c.function, tr, c.solver,
                       c.eval_points, keyspace)
        return c

    def to_connection(self):
        """Create a standard Nengo connection from this object.
        """
        return nengo.Connection(self.pre, self.post, synapse=self.synapse,
                                function=self.function,
                                transform=self.transform, solver=self.solver,
                                eval_points=self.eval_points,
                                add_to_container=False)


def remove_passthrough_nodes(objs, connections):  # noqa: C901
    """Returns a version of the model without passthrough Nodes

    For some backends (such as SpiNNaker), it is useful to remove Nodes that
    have 'None' as their output.  These nodes simply sum their inputs and
    use that as their output. These nodes are defined purely for organizational
    purposes and should not affect the behaviour of the model.  For example,
    the 'input' and 'output' Nodes in an EnsembleArray, which are just meant to
    aggregate data.

    Note that removing passthrough nodes can simplify a model and may be useful
    for other backends as well.  For example, an EnsembleArray connected to
    another EnsembleArray with an identity matrix as the transform
    should collapse down to D Connections between the corresponding Ensembles
    inside the EnsembleArrays.

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the objects in the model
    connections : list of Connections
        All the Connections in the model

    Returns the objs and connections of the resulting model.  The passthrough
    Nodes will be removed, and the Connections that interact with those Nodes
    will be replaced with equivalent Connections that don't interact with those
    Nodes.
    """

    inputs, outputs = find_all_io(connections)
    result_conn = list(connections)
    result_objs = list(objs)

    # look for passthrough Nodes to remove
    for obj in objs:
        if isinstance(obj, nengo.Node) and obj.output is None:
            result_objs.remove(obj)

            # get rid of the connections to and from this Node
            for c in inputs[obj]:
                result_conn.remove(c)
                outputs[c.pre].remove(c)
            for c in outputs[obj]:
                result_conn.remove(c)
                inputs[c.post].remove(c)

            # replace those connections with equivalent ones
            for c_in in inputs[obj]:
                if c_in.pre is obj:
                    raise Exception('Cannot remove a Node with feedback')

                for c_out in outputs[obj]:
                    c = _create_replacement_connection(c_in, c_out)
                    if c is not None:
                        result_conn.append(c)
                        # put this in the list, since it might be used
                        # another time through the loop
                        outputs[c.pre].append(c)
                        inputs[c.post].append(c)

    return result_objs, result_conn


def find_all_io(connections):
    """Build up a list of all inputs and outputs for each object"""
    inputs = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for c in connections:
        inputs[c.post].append(c)
        outputs[c.pre].append(c)
    return inputs, outputs


def _create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post is c_out.pre
    assert c_in.post.output is None

    # determine the filter for the new Connection
    if c_in.synapse is None:
        synapse = c_out.synapse
    elif c_out.synapse is None:
        synapse = c_in.synapse
    else:
        raise NotImplementedError('Cannot merge two filters')
        # Note: the algorithm below is in the right ballpark,
        #  but isn't exactly the same as two low-pass filters
        # filter = c_out.filter + c_in.filter

    function = c_in.function
    if c_out.function is not None:
        raise Exception('Cannot remove a Node with a '
                        'function being computed on it')

    # compute the combined transform
    transform = np.dot(full_transform(c_out), full_transform(c_in))
    # check if the transform is 0 (this happens a lot
    #  with things like identity transforms)
    if np.all(transform == 0):
        pass

    # Determine the combined keyspace
    if (getattr(c_out, 'keyspace', None) is not None and
            getattr(c_in, 'keyspace', None) is None):
        # If the out keyspace is specified but the IN isn't, then use the out
        # keyspace
        keyspace = getattr(c_out, 'keyspace', None)
    elif (getattr(c_in, 'keyspace', None) is not None and
            getattr(c_out, 'keyspace', None) is None):
        # Vice versa
        keyspace = getattr(c_in, 'keyspace', None)
    elif getattr(c_in, 'keyspace', None) == getattr(c_out, 'keyspace', None):
        # The keyspaces are equivalent
        keyspace = getattr(c_in, 'keyspace', None)
    else:
        # XXX: The incoming and outcoming connections have assigned
        #      keyspaces, this shouldn't occur (often if not at all).
        raise NotImplementedError('Cannot merge two keyspaces.')

    # Determine the type of connection to use
    if c_in.__class__ is c_out.__class__:
        # Types are equivalent, so use this type
        ctype = c_in.__class__
    elif c_in.__class__ is IntermediateConnection:
        # In type is default, use out type
        ctype = c_out.__class__
    elif c_out.__class__ is IntermediateConnection:
        # Out type is default, use in type
        ctype = c_in.__class__
    else:
        raise NotImplementedError("Cannot merge '%s' and '%s' connection "
                                  "types." % (c_in.__class__,
                                              c_out.__class__))

    if ctype is nengo.Connection:
        ctype = IntermediateConnection

    c = ctype(c_in.pre, c_out.post, synapse=synapse, transform=transform,
              function=function, keyspace=keyspace)
    return c
