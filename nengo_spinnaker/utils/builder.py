import numpy as np

import nengo
from nengo.utils.builder import full_transform

from ..connection import IntermediateConnection


def create_replacement_connection(c_in, c_out):
    """Generate a new Connection to replace two through a passthrough Node"""
    assert c_in.post_obj is c_out.pre_obj
    assert c_in.post_obj.output is None

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
        return None

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

    c = ctype(c_in.pre_obj, c_out.post_obj, synapse=synapse,
              transform=transform, function=function, keyspace=keyspace)
    return c
