"""Builder for running Nengo models on SpiNNaker

Converts a Nengo model (rather, Network) into the graph based representation
required by PACMAN.
"""

import inspect
import re

import nengo
import nengo.utils.builder

from pacman103.core import dao


class Builder(object):
    """Converts a Nengo model into a PACMAN appropriate data structure."""

    def __init__(self):
        # Build by inspection the diction mapping of things we can build
        builds = filter(lambda m: "_build_" == m[0][0:7],
                        inspect.getmembers(self, inspect.ismethod))
        objects = dict(inspect.getmembers(nengo, inspect.isclass))
        self.builders = dict()

        for (s, f) in builds:
            obj_name = re.sub(
                r'_(\w)', lambda m: m.group(1).upper(), re.sub('_build', '', s)
            )
            if obj_name in objects:
                self.builders[objects[obj_name]] = f

    def _build(self, obj):
        """Call the appropriate build function for the given object."""
        if not type(obj) in self.builders:
            raise TypeError("Cannot build a '%s' object." % type(obj))
        self.builders[type(obj)](obj)

    def __call__(self, model, dt):
        """Return a PACMAN103 DAO containing a representation of the given
        model, and a list of I/O Nodes with references to their connected Rx
        and Tx components.
        """
        # Create a DAO to store PACMAN data and Node list for the simulator
        self.dao = dao.DAO("nengo")
        self.nodes = list()

        # Get a new network structure with passthrough nodes removed
        (objs, connections) = nengo.utils.builder.remove_passthrough_nodes(
            model.objs, model.connections
        )

        # Build each of the objects
        for obj in objs:
            self._build(obj)

        # Build each of the connections
        for conn in connections:
            self._build(conn)

    def _build_ensemble(self, ens):
        # Add an appropriate Vertex which deals with the Ensemble
        raise NotImplementedError

    def _build_node(self, node):
        # Manage Rx and Tx components
        # Add the Node to the node list
        raise NotImplementedError

    def _build_connection(self, conn):
        # Add appropriate Edges between Vertices
        # In the case of Nodes, determine which Rx and Tx components we need
        # to connect to.
        raise NotImplementedError
