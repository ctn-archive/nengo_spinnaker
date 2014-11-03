import random

from .. import connections as connections_utils


def test_replace_object_in_connections():
    """Test that connections are replaced with new connections when they
    originate or terminate with objects which are to be replaced.
    """
    class FalseConnection(object):
        def __init__(self, pre_obj, post_obj):
            self.pre_obj = pre_obj
            self.post_obj = post_obj

        def __repr__(self):
            return "FalseConnection({}, {})".format(self.pre_obj,
                                                    self.post_obj)

    class FalseEnsemble(object):
        def __repr__(self):
            return '<FalseEnsemble at {:#x}>'.format(id(self))

    class FalseIntermediateEnsemble(object):
        def __repr__(self):
            return '<FalseIntermediateEnsemble at {:#x}>'.format(id(self))

    # Create some non-replaced objects and some replaced objects
    non_replaced_obj = [FalseEnsemble() for n in range(2)]
    replaced_objs = {FalseEnsemble(): FalseIntermediateEnsemble() for n in
                     range(3)}

    # Create a complete set of connections between these sets of objects,
    # randomly remove some elements.
    connections = list()
    connections.extend(FalseConnection(nro, ro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    connections.extend(FalseConnection(ro, nro) for nro in non_replaced_obj for
                       ro in replaced_objs.keys())
    random.shuffle(connections)
    connections = connections[:-3]

    # Add one connection which will remain the same
    connections.append(
        FalseConnection(non_replaced_obj[0], non_replaced_obj[1]))

    # Replace connections where objects have also been replaced
    new_connections = connections_utils.\
        replace_objects_in_connections(connections, replaced_objs)

    # Check for success
    assert len(connections) == len(new_connections)
    assert connections[-1] in new_connections

    for (oc, nc) in zip(connections[:-1], new_connections):
        assert nc is not oc

        if oc.pre_obj in replaced_objs:
            assert nc.pre_obj is replaced_objs[oc.pre_obj]
        if oc.post_obj in replaced_objs:
            assert nc.post_obj is replaced_objs[oc.post_obj]
