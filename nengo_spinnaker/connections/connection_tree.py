import collections
import copy
from six import iteritems, itervalues

from ..spinnaker.edges import Edge


class ConnectionTree(object):
    """A representation of the connectivity of a Nengo network.

    The connection "tree" consists of a set of root nodes, each of which
    represents an object which transmits data to other objects.  The first set
    of branches represents outgoing connections from each object, these will be
    automatically merged and reduced.  The final lists of leaves represent each
    object, port and filter which receives the data transmitted over each
    connection.

    For example, the model:

        (a) --CONN1-> (b)
         | \           |
         |  \-CONN1----/
         |
         \--CONN2---> (c)

    Will result in a "forest":

        a -----> (CONN1) -----> b.INPUT
          \              \----> b.INPUT
           \---> (CONN2) -----> c.INPUT

    Meaning that each packet `a' transmits over CONN1 will be "received
    twice" by `b', packets transmitted by `a' over CONN2 will only be
    received by `c'.
    """
    def __init__(self, connection_tree):
        """Create a new ConnectionTree from the specific dictionary object.

        The dictionary object should be a mapping of objects to a mapping of
        outgoing connections to lists of incoming connections (i.e., `obj ->
        (outgoing connections -> [incoming connections])`.

        A connection tree is intended to be immutable, any new changes will
        result in the creation of a new instance of the tree.
        """
        # Store a reference
        self._connectivity_tree = connection_tree

    @classmethod
    def from_intermediate_connections(cls, connections):
        """Construct a new ConnectionTree from connection objects.

        Connection objects are required to provide two methods:
         - `get_reduced_outgoing_connection` which provides an outgoing
           connection element.
         - `get_reduced_incoming_connection` which provides an incoming
           connection element.
        """
        connectivity_tree = _make_empty_connectivity_tree()

        for c in connections:
            # Get the reduced outgoing connection for this object, the reduced
            # incoming connection and the originating object; add these to the
            # tree.
            (obj, outgoing, incoming) = (
                c.pre_obj, c.get_reduced_outgoing_connection(),
                c.get_reduced_incoming_connection()
            )

            # Add the object, outgoing connection and incoming connection to
            # the tree.
            connectivity_tree[obj][outgoing].append(incoming)

        # Having built the connectivity tree dictionary we call the usual
        # constructor.
        return cls(connectivity_tree)

    def get_originating_objects(self):
        """Get a list of objects which from which connections originate.
        """
        # Originating objects are simply the keys of the dictionary
        return list(self._connectivity_tree.keys())

    def get_terminating_objects(self):
        """Get a list of objects at which connections terminate.
        """
        # Terminating objects are referenced in the leaves, so we perform a
        # leaf walk.
        terminators = set()
        for oconns in itervalues(self._connectivity_tree):
            for iconns in itervalues(oconns):
                for c in iconns:
                    terminators.add(c.target.target_object)

        return list(terminators)

    def get_objects(self):
        """Get a list of all objects which are present in the tree.
        """
        # This is not the most efficient implementation of this, but it is the
        # cleanest.  Neaten up if we appear to be spending lots of time here
        # (probably unlikely).
        return list(set(self.get_originating_objects()) |
                    set(self.get_terminating_objects()))

    def get_outgoing_connections(self, obj):
        """Get a list of all outgoing connections for the given object.
        """
        # Outgoing connections from an object are retrieved from the mapping.
        # If the requested object has no outgoing connections then return the
        # empty list.
        if obj not in self._connectivity_tree:
            return list()

        return list(self._connectivity_tree[obj].keys())

    def get_incoming_connections(self, obj):
        """Get a map of incoming connections to their sources for the object.

        The map return is from port to filters to keyspaces which feed in
        (i.e., `port -> (filter -> [keyspace])`.
        """
        # Maps port -> (filter -> [keyspace])
        incs = collections.defaultdict(lambda: collections.defaultdict(list))

        # Leaf-walk for incoming edges and add entries that correspond to this
        # object to the dictionary.
        for oconns in itervalues(self._connectivity_tree):
            for (oconn, iconns) in iteritems(oconns):
                for c in iconns:
                    # If this incoming connection corresponds to the object
                    # we're looking for
                    if c.target.target_object is obj:
                        # Then add the port, filter and corresponding keyspace
                        # to the tree of dictionary of incoming connections.
                        incs[c.target.port][c.filter_object].append(
                            oconn.keyspace)

        return incs

    def get_new_tree_with_replaced_objects(self, replacements,
                                           replace_when_originating=True,
                                           replace_when_terminating=True):
        """Return a new tree with specified objects replaced.

        :param dict replacements: A dictionary mapping old to new objects.
        :param bool replace_when_originating: Replace objects when they are
            at the start of connections.
        :param bool replace_when_terminating: Replace objects when they are
            at the end of connections.
        :rtype: :py:class:`ConnectionTree`
        """
        # Perform a shallow copy of the tree, but with some objects replaced.
        # Walk through the tree, copying and replacing as required.

        # First, create a function we can query for the correct version of an
        # object to use.
        def repl(obj):
            """Replace an object, or keep it the same."""
            if obj not in replacements:
                return obj
            return replacements[obj]

        # Now construct the new tree by walking through the old one.
        connectivity_tree = _make_empty_connectivity_tree()
        for (obj, outgoing_conns) in iteritems(self._connectivity_tree):
            # Get the new object to use
            if replace_when_originating:
                new_obj = repl(obj)
            else:
                new_obj = obj

            # Copy each outgoing connection and add to the tree.
            for (out_conn, in_conns) in iteritems(outgoing_conns):
                out_conn = copy.copy(out_conn)

                # Copy each incoming connection, and add to the tree, replacing
                # the target object as required.
                for in_conn in in_conns:
                    # Copy and replace.
                    in_conn = copy.copy(in_conn)
                    if replace_when_terminating:
                        in_conn.target.target_object = repl(
                            in_conn.target.target_object)

                    # Add the new incoming connection.
                    connectivity_tree[new_obj][out_conn].append(in_conn)

        # Return a new connectivity tree
        return self.__class__(connectivity_tree)

    def get_new_tree_with_applied_keyspace(self, default_keyspace):
        """Return a new tree with keyspaces applied to outgoing connections.

        :rtype: :py:class:`ConnectionTree`
        """
        # Walk through the tree, copying and filling in keyspaces as required.
        connectivity_tree = _make_empty_connectivity_tree()
        for o, (obj, oconns) in enumerate(iteritems(self._connectivity_tree)):
            # Get the object keyspace
            object_keyspace = default_keyspace(o=o)

            # Copy each outgoing connection and add to the tree.
            for i, (out_conn, in_conns) in enumerate(iteritems(oconns)):
                out_conn = copy.copy(out_conn)

                # Fill in the keyspace for the connection
                out_conn.keyspace = object_keyspace(i=i)

                # Copy in the incoming keyspaces
                for in_conn in in_conns:
                    in_conn = copy.copy(in_conn)
                    connectivity_tree[obj][out_conn].append(in_conn)

        # Return a new connectivity tree
        return self.__class__(connectivity_tree)

    def get_folded_edges(self):
        """Return a list of edges representing the connectivity of the tree.

        :returns: A list of :py:class:`Edge` objects.
        """
        edges = list()

        # For each originating object
        for pre_obj, outgoing_conns in iteritems(self._connectivity_tree):
            # For each outgoing connection
            for outgoing_conn, incoming_conns in iteritems(outgoing_conns):
                # For each incoming connection
                for incoming_conn in incoming_conns:
                    # Create a new edge
                    post_obj = incoming_conn.target.target_object
                    keyspace = outgoing_conn.keyspace
                    edges.append(Edge(pre_obj, post_obj, keyspace))

        # Return the list of edges
        return edges


def _make_empty_connectivity_tree():
    # Creates a dictionary representing the connectivity of the network.
    # Maps: object -> (outgoing connection -> [incoming_connections])
    return collections.defaultdict(lambda: collections.defaultdict(list))


def _pretty_print(tree):  # pragma: no cover
    """Pretty print a connection tree.
    """
    for (obj, oconns) in iteritems(tree._connectivity_tree):
        print("{}\n\t\t|".format(obj))

        iter_oconns = iteritems(oconns)
        last = False

        try:
            (oconn, iconns) = iter_oconns.next()
        except StopIteration:
            last = True

        while not last:
            try:
                (oconn_next, iconns_next) = iter_oconns.next()
            except StopIteration:
                last = True

            print("\t\t{}-> {}".format("\\" if last else "|", oconn))

            for ic in iconns:
                print(
                    "\t\t{}\t\t{}-> {}".format(
                        "" if last else "|",
                        "\\" if ic is iconns[-1] else "+",
                        ic
                    )
                )

            (oconn, iconns) = (oconn_next, iconns_next)
