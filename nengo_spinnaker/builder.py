"""Builder for running Nengo models on SpiNNaker

Converts a Nengo model (rather, Network) into the graph based representation
required by PACMAN.
"""

import inspect
import re
import numpy as np
import itertools

import nengo
import nengo.utils.builder

from pacman103.core import dao
from pacman103.front import common
from pacman103.lib import graph

from . import edges
from . import ensemble_vertex

edge_builders = {}


def register_build_edge(pre=None, post=None):
    def f_(f):
        global edge_builders

        edge_builders[(pre, post)] = f
        return f
    return f_


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
        for obj_class in obj.__class__.__mro__:
            if obj_class in self.builders:
                self.builders[obj_class](obj)
                break
        else:
            raise TypeError("Cannot build a '%s' object." % type(obj))

    def __call__(self, model, dt, seed=None, node_builder=None):
        """Return a PACMAN103 DAO containing a representation of the given
        model, and a list of I/O Nodes with references to their connected Rx
        and Tx components.
        """
        self.rng = np.random.RandomState(seed)

        # Create a DAO to store PACMAN data and Node list for the simulator
        self.dao = dao.DAO("nengo")
        self.ensemble_vertices = dict()
        self.nodes = list()
        self.node_node_connections = list()

        # Store a Node Builder
        self.node_builder = node_builder

        # Get a new network structure with passthrough nodes removed
        (objs, connections) = nengo.utils.builder.remove_passthrough_nodes(
            *nengo.utils.builder.objs_and_connections(model)
        )

        # Create a MultiCastVertex
        self._mc_tx_vertex = None

        # Build each of the objects
        for obj in objs:
            self._build(obj)

        # Build each of the connections
        for conn in connections:
            self._build(conn)

        # Return the DAO
        return self.dao, self.nodes, self.node_node_connections

    def add_vertex(self, vertex):
        self.dao.add_vertex(vertex)

    def add_edge(self, edge):
        self.dao.add_edge(edge)

    def _build_ensemble(self, ens):
        # Add an appropriate Vertex which deals with the Ensemble
        vertex = ensemble_vertex.EnsembleVertex(ens, self.rng)
        self.add_vertex(vertex)
        self.ensemble_vertices[ens] = vertex

    def _build_node(self, node):
        if hasattr(node, "spinnaker_build"):
            node.spinnaker_build(self)
        else:
            self.nodes.append(node)
            self.node_builder.build_node(self, node)

    def _build_connection(self, c):
        # Add appropriate Edges between Vertices
        # Determine which edge building function to use
        # TODO Modify to fallback to `isinstance` where possible
        edge = None

        pre_c = c.pre.__class__.__mro__
        post_c = c.post.__class__.__mro__

        for key in itertools.chain(*[[(a, b) for b in post_c] for a in pre_c]):
            if key in edge_builders:
                edge = edge_builders[key](self, c)
                break
        else:
            raise TypeError("Cannot connect '%s' -> '%s'" % (
                type(c.pre), type(c.post)))

        if edge is not None:
            self.add_edge(edge)

    def connect_to_multicast_vertex(self, postvertex):
        """Create a connection from the MultiCastVertex to the given
        postvertex.

        The postvertex must support the method `get_commands`.
        """
        if self._mc_tx_vertex is None:
            self._mc_tx_vertex = common.MultiCastSource()
            self.add_vertex(self._mc_tx_vertex)

        self.add_edge(graph.Edge(self._mc_tx_vertex, postvertex))

    def get_node_in_vertex(self, c):
        """Get the Vertex for input to the terminating Node of the given
        Connection
        """
        return self.node_builder.get_node_in_vertex(self, c)

    def get_node_out_vertex(self, c):
        """Get the Vertex for output from the originating Node of the given
        Connection"""
        return self.node_builder.get_node_out_vertex(self, c)


@register_build_edge(pre=nengo.Ensemble, post=nengo.Ensemble)
def _ensemble_to_ensemble(builder, c):
    prevertex = builder.ensemble_vertices[c.pre]
    postvertex = builder.ensemble_vertices[c.post]
    edge = edges.DecoderEdge(c, prevertex, postvertex)
    edge.index = prevertex.decoders.get_decoder_index(edge)
    postvertex.filters.add_edge(edge)
    return edge


@register_build_edge(pre=nengo.Ensemble, post=nengo.Node)
def _ensemble_to_node(builder, c):
    # Get the vertices
    prevertex = builder.ensemble_vertices[c.pre]
    postvertex = builder.get_node_in_vertex(c)

    # Create the edge
    edge = edges.DecoderEdge(c, prevertex, postvertex)
    edge.index = prevertex.decoders.get_decoder_index(edge)

    # Store the filter value if necessary
    if hasattr(postvertex, "filters"):
        postvertex.filters.add_edge(edge)

    return edge


@register_build_edge(pre=nengo.Node, post=nengo.Ensemble)
def _node_to_ensemble(builder, c):
    # If the Node has constant output then add to the direct input
    # for the Ensemble and don't add an edge, otherwise add an
    # edge from the appropriate Rx element to the Ensemble.
    postvertex = builder.ensemble_vertices[c.post]
    if c.pre.output is not None and not callable(c.pre.output):
        postvertex.direct_input += np.asarray(c.pre.output)
    else:
        prevertex = builder.get_node_out_vertex(c)
        edge = edges.InputEdge(c, prevertex, postvertex,
                               filter_is_accumulatory=False)
        postvertex.filters.add_edge(edge)
        return edge


@register_build_edge(pre=nengo.Node, post=nengo.Node)
def _node_to_node(builder, c):
    builder.node_node_connections.append(c)
