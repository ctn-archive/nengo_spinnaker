"""Builder for running Nengo models on SpiNNaker

Converts a Nengo model (rather, Network) into the graph based representation
required by PACMAN.
"""

import inspect
import re
import numpy as np
import itertools
import collections

import nengo
import nengo.utils.builder

from pacman103.core import dao
from pacman103.front import common
from pacman103.lib import graph

from . import edges
from . import ensemble_vertex
from . import filter_vertex
from . import receive_vertex
from . import serial_vertex
from . import transmit_vertex

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

    def __call__(self, model, dt, seed=None, use_serial=False):
        """Return a PACMAN103 DAO containing a representation of the given
        model, and a list of I/O Nodes with references to their connected Rx
        and Tx components.
        """
        self.rng = np.random.RandomState(seed)

        # Create a DAO to store PACMAN data and Node list for the simulator
        self.dao = dao.DAO("nengo")
        self.dao.ensemble_vertices = self.ensemble_vertices = dict()
        self.dao.tx_vertices = self._tx_vertices = list()
        self.dao.tx_assigns = self._tx_assigns = dict()
        self.dao.rx_vertices = self._rx_vertices = list()
        self.dao.rx_assigns = self._rx_assigns = dict()
        self.dao.node_to_node_edges = self._node_to_node_edges = list()

        # Add a serial vertex if required
        self.use_serial = use_serial
        self.serial = None
        if use_serial:
            self.serial = serial_vertex.SerialVertex()
            self.dao.add_vertex(self.serial)

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

        if self.use_serial:
            self.serial_rx = connections.defaultdict(list)
            self.serial_tx = {}
            for edge in self.serial.in_edges:
                for subedge in edge.subedges:
                    key = edge.prevertex.generate_routing_info(subedge)[0]
                    node = edge.post
                    self.serial_tx[key] = node

            for edge in self.serial.out_edges:
                for subedge in edge.subedges:
                    key = edge.prevertex.generate_routing_info(subedge)[0]
                    node = edge.pre
                    self.serial_rx[node].append(key)

        # Return the DAO
        return self.dao

    def add_vertex(self, vertex):
        self.dao.add_vertex(vertex)

    def add_edge(self, edge):
        self.dao.add_edge(edge)

    def _build_ensemble(self, ens):
        # Add an appropriate Vertex which deals with the Ensemble
        vertex = ensemble_vertex.EnsembleVertex(ens, self.rng)
        self.dao.add_vertex(vertex)
        self.ensemble_vertices[ens] = vertex

    def _build_node(self, node):
        # If the Node has a `spinnaker_build` function then ask it for a vertex
        if hasattr(node, "spinnaker_build"):
            node.spinnaker_build(self)
            return

        # Otherwise the node is assigned to Rx and Tx components as required
        # If the Node has input, then assign the Node to a Tx component
        if node.size_in > 0 and not self.use_serial:
            # Try to fit the Node in an existing Tx Element
            # Most recently added Txes are nearer the start
            tx_assigned = False
            for tx in self._tx_vertices:
                if tx.remaining_dimensions >= node.size_in:
                    tx.assign_node(node)
                    tx_assigned = True
                    self._tx_assigns[node] = tx
                    break

            # Otherwise create a new Tx element
            if not tx_assigned:
                tx = transmit_vertex.TransmitVertex(
                    label="Tx%d" % len(self._tx_vertices)
                )
                self.dao.add_vertex(tx)
                tx.assign_node(node)
                self._tx_assigns[node] = tx
                self._tx_vertices.insert(0, tx)

        # If the Node has output, and that output is not constant, then assign
        # the Node to an Rx component.
        if node.size_out > 0 and callable(node.output) and not self.use_serial:
            # Try to fit the Node in an existing Rx Element
            # Most recently added Rxes are nearer the start
            rx_assigned = False
            for rx in self._rx_vertices:
                if rx.remaining_dimensions >= node.size_out:
                    rx.assign_node(node)
                    rx_assigned = True
                    self._rx_assigns[node] = rx
                    break

            # Otherwise create a new Rx element
            if not rx_assigned:
                rx = receive_vertex.ReceiveVertex(
                    label="Rx%d" % len(self._rx_vertices)
                )
                self.dao.add_vertex(rx)
                rx.assign_node(node)
                self._rx_assigns[node] = rx
                self._rx_vertices.insert(0, rx)

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
            self.dao.add_edge(edge)

        def connect_to_multicast_vertex(self, postvertex):
            """Create a connection from the MultiCastVertex to the given
            postvertex.

            The postvertex must support the method `get_commands`.
            """
            if self._mc_tx_vertex is None:
                self._mc_tx_vertex = common.MultiCastSource()
                self.dao.add_vertex(self._mc_tx_vertex)

            self.dao.add_edge(graph.Edge(self._mc_tx_vertex, postvertex))


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
    prevertex = builder.ensemble_vertices[c.pre]

    if builder.use_serial:
        postvertex = filter_vertex.FilterVertex(c.post.size_in,
            output_id=0, update_period=10)
        builder.add_vertex(postvertex)
        edge = edges.NengoEdge(c, postvertex, builder.serial)
        builder.add_edge(edge)
    else:
        postvertex = builder._tx_assigns[c.post]

    edge = edges.DecoderEdge(c, prevertex, postvertex)
    edge.index = prevertex.decoders.get_decoder_index(edge)
    if builder.use_serial:
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
        if builder.use_serial:
            prevertex = builder.serial
        else:
            prevertex = builder._rx_assigns[c.pre]
        edge = edges.InputEdge(c, prevertex, postvertex)
        postvertex.filters.add_edge(edge)
        return edge


@register_build_edge(pre=nengo.Node, post=nengo.Node)
def _node_to_node(builder, c):
    builder._node_to_node_edges.append(c)
