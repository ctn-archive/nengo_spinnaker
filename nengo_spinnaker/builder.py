"""Builder for running Nengo models on SpiNNaker

Converts a Nengo model (rather, Network) into the graph based representation
required by PACMAN.
"""

import inspect
import re
import numpy as np

import nengo
import nengo.utils.builder

from pacman103.core import dao

from . import decoder_edge
from . import ensemble_vertex
from . import input_edge


class Builder(object):
    """Converts a Nengo model into a PACMAN appropriate data structure."""

    def __init__(self):
        # Build by inspection the dictionary of things we can build
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

    def __call__(self, model, dt, seed=None):
        """Return a PACMAN103 DAO containing a representation of the given
        model, and a list of I/O Nodes with references to their connected Rx
        and Tx components.
        """
        self.rng = np.random.RandomState(seed)

        # Create a DAO to store PACMAN data and Node list for the simulator
        self.dao = dao.DAO("nengo")
        self.ensemble_vertices = dict()  # Map of Ensembles to their vertices
        self._tx_vertices = list()
        self._rx_vertices = list()
        self._node_to_node_edges = list()

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
        vertex = ensemble_vertex.EnsembleVertex(ens, self.rng)
        self.dao.add_vertex(vertex)
        self.ensembles_vertices[ens] = vertex

    def _build_node(self, node):
        # Manage Rx and Tx components
        # Add the Node to the node list
        raise NotImplementedError

    def _build_connection(self, c):
        # Add appropriate Edges between Vertices
        # In the case of Nodes, determine which Rx and Tx components we need
        # to connect to.
        if isinstance(c.pre, nengo.Ensemble):
            prevertex = self.ensembles_vertices(c.pre)
            if isinstance(c.post, nengo.Ensemble):
                # Ensemble -> Ensemble
                postvertex = self.ensembles_vertices(c.post)
                self.dao.add_edge(
                    decoder_edge.DecoderEdge(c, prevertex, postvertex)
                )
            elif isinstance(c.post, nengo.Node):
                # Ensemble -> Node
                postvertex = self._tx_vertices(c.post)
                self.dao.add_edge(
                    decoder_edge.DecoderEdge(c, prevertex, postvertex)
                )
        elif isinstance(c.pre, nengo.Node):
            if isinstance(c.post, nengo.Ensemble):
                # Node -> Ensemble
                # If the Node has constant output then add to the direct input
                # for the Ensemble and don't add an edge, otherwise add an
                # edge from the appropriate Rx element to the Ensemble.
                postvertex = self.ensembles_vertices(c.post)
                if c.pre.output is not None and not callable(c.pre.output):
                    postvertex.direct_input += np.asarray(c.pre.output)
                else:
                    prevertex = self._rx_vertices(c.pre)
                    self.dao.add_edge(
                        input_edge.InputEdge(c, prevertex, postvertex)
                    )
            elif isinstance(c.post, nengo.Node):
                # Node -> Node
                self._node_to_node_edges.append(c)
