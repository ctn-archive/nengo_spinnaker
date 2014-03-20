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

from . import edges
from . import ensemble_vertex, transmit_vertex, receive_vertex


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
        self.dao.ensemble_vertices = self.ensemble_vertices = dict()
        self.dao.tx_vertices = self._tx_vertices = list()
        self.dao.tx_assigns = self._tx_assigns = dict()
        self.dao.rx_vertices = self._rx_vertices = list()
        self.dao.rx_assigns = self._rx_assigns = dict()
        self.dao.node_to_node_edges = self._node_to_node_edges = list()

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

        # Return the DAO
        return self.dao

    def _build_ensemble(self, ens):
        # Add an appropriate Vertex which deals with the Ensemble
        vertex = ensemble_vertex.EnsembleVertex(ens, self.rng)
        self.dao.add_vertex(vertex)
        self.ensemble_vertices[ens] = vertex

    def _build_node(self, node):
        # If the Node has input, then assign the Node to a Tx component
        if node.size_in > 0:
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
        if node.size_out > 0 and callable(node.output):
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
        # In the case of Nodes, determine which Rx and Tx components we need
        # to connect to.
        if isinstance(c.pre, nengo.Ensemble):
            prevertex = self.ensemble_vertices[c.pre]
            edge = None
            if isinstance(c.post, nengo.Ensemble):
                # Ensemble -> Ensemble
                postvertex = self.ensemble_vertices[c.post]
                edge = edges.DecoderEdge(c, prevertex, postvertex)
            elif isinstance(c.post, nengo.Node):
                # Ensemble -> Node
                postvertex = self._tx_assigns[c.post]
                edge = edges.DecoderEdge(c, prevertex, postvertex)
            else:
                raise TypeError(
                    "Cannot connect an Ensemble to a '%s'" % type(c.post)
                )
            edge.index = prevertex.decoders.get_decoder_index(edge)
            self.dao.add_edge(edge)

        elif isinstance(c.pre, nengo.Node):
            if isinstance(c.post, nengo.Ensemble):
                # Node -> Ensemble
                # If the Node has constant output then add to the direct input
                # for the Ensemble and don't add an edge, otherwise add an
                # edge from the appropriate Rx element to the Ensemble.
                postvertex = self.ensemble_vertices[c.post]
                if c.pre.output is not None and not callable(c.pre.output):
                    postvertex.direct_input += np.asarray(c.pre.output)
                else:
                    prevertex = self._rx_assigns[c.pre]
                    self.dao.add_edge(
                        edges.InputEdge(c, prevertex, postvertex)
                    )
            elif isinstance(c.post, nengo.Node):
                # Node -> Node
                self._node_to_node_edges.append(c)
            else:
                raise TypeError(
                    "Cannot connect an Ensemble to a '%s'" % type(c.post)
                )

        else:
            raise TypeError(
                "Cannot start a connection from a '%s'" % type(c.pre)
            )
