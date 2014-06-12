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
from .nodes import value_source_vertex
from .utils import connections, probes
from . import value_sink_vertex

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

    def __call__(self, model, dt, seed=None, node_builder=None, config=None):
        """Return a PACMAN103 DAO containing a representation of the given
        model, and a list of Nodes and list of Node->Node connections.

        :param model: the Nengo model to build
        :param dt: timestep to use in simulation (e.g., 0.001)
        :param seed: seed for random number generators
        :param node_builder: a builder for constructing the IO required by
                             Nodes
        :param config: a Config option for SpiNNaker specific object
                       configuration

        :returns: a 4-tuple of a DAO, list of Nodes, list of Node->Node
                  connections, list of probes
        """
        self.rng = np.random.RandomState(seed)
        self.dt = dt

        # Create a DAO to store PACMAN data and Node list for the simulator
        self.dao = dao.DAO("nengo")
        self.ensemble_vertices = dict()
        self.nodes = list()
        self.node_node_connections = list()
        self.f_of_t_vertices = dict()
        self.probes = list()

        # Store a Node Builder
        self.node_builder = node_builder

        # Store the Config (create an empty one if None)
        if config is None:
            from .config import Config
            config = Config()
        self.config = config

        # Get a new network structure with passthrough nodes removed
        (objs, connections) = nengo.utils.builder.remove_passthrough_nodes(
            *nengo.utils.builder.objs_and_connections(model)
        )

        # Create a MultiCastVertex
        self._mc_tx_vertex = None

        # Build each of the objects
        for obj in objs:
            self._build(obj)
        
        # Filter connections, removing modulatory connections as, on SpiNNaker,
        # These may actually need to be reconnected to the pre end of modulated 
        # Decoder learning connections as this is where the decoders are!
        connections = filter(lambda c: c.modulatory == False, connections)
       
        # Perform any learning-rule specific rerouting of modulatory connections
        connections = self._reroute_modulatory_connections(connections)
        
        # Build each of the connections
        for conn in connections:
            self._build(conn)

        # Probes
        for probe in model.probes:
            if isinstance(probe.target, nengo.Ensemble):
                vertex = self.ensemble_vertices[probe.target]

                if probe.attr == 'spikes':
                    vertex.record_spikes = True
                    self.probes.append(probes.SpikeProbe(vertex, probe))
                elif probe.attr == 'decoded_output':
                    postvertex = value_sink_vertex.ValueSinkVertex(
                        probe.size_in)
                    self.add_vertex(postvertex)
                    self.add_edge(
                        edges.ValueProbeEdge(probe, vertex, postvertex,
                                             size_in=vertex._ens.size_out,
                                             size_out=vertex._ens.size_out))
                    self.probes.append(
                        probes.DecodedValueProbe(vertex, postvertex, probe))
                else:
                    raise NotImplementedError(
                        "Cannot probe '%s' on Ensembles" % probe.attr)
            else:
                raise NotImplementedError(
                    "Cannot probe '%s' objects" % type(probe.target))

        # Return the DAO, Nodes, Node->Node connections and Probes
        return self.dao, self.nodes, self.node_node_connections, self.probes

    def add_vertex(self, vertex):
        self.dao.add_vertex(vertex)

    def add_edge(self, edge):
        self.dao.add_edge(edge)

    def _reroute_modulatory_connections(self, connections):
        # Create new connections list, initially just containing 
        # All the original, non-modulatory connections
        new_connections = connections
        
        # Loop through all connections and learning rules associated with them
        for connection in connections:
            for learning_rule in connection.learning_rule:
                # **HACK** using the _build mechanism here would be much nicer
                # If learning rule is PES
                if isinstance(learning_rule, nengo.PES):
                    # If the pre-connection object (where the deocoder is)
                    # Isn't in the ensemble vertices dictionary, throw
                    if connection.pre not in self.ensemble_vertices:
                        raise TypeError("Object %s, on the pre-side of connection\
                            %s is not an ensemble so does not support PES" 
                            % (connection.pre, connection))
                    
                    # Cache refence to PES rule in pre-connection ensemble
                    # **TODO** API probably should look more like add_learning_rule
                    pre_ensemble = self.ensemble_vertices[connection.pre]
                    pre_ensemble.pes = learning_rule
                    
                    # Create copy of error connection, connecting source of error
                    # To the pre-connection object (where the decoder is)
                    # **YUCK** is there a nicer way of doing this?
                    error_con = learning_rule.error_connection
                    pre_error_con = nengo.Connection(error_con.pre, connection.pre, 
                                                        synapse = error_con.synapse,
                                                        transform = error_con.transform,
                                                        solver = error_con.solver,
                                                        function = error_con.function,
                                                        modulatory = error_con.modulatory,
                                                        eval_points = error_con.eval_points,
                                                        learning_rule = error_con.learning_rule,
                                                        seed = error_con.seed)
                    new_connections.append(pre_error_con)
                
                # Otherwise (unsupported learning rule)
                else:
                    raise NotImplementedError(
                        "Learning rules of type %s are not supported" % type(learning_rule))
        
        return new_connections
    
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
            if self.config[node].f_of_t:
                # Node is a function of time to be evaluated in advance
                pass
            else:
                # Nodes to be executed on the host
                self.node_builder.build_node(self, node)

    def _build_connection(self, c):
        # Add appropriate Edges between Vertices
        # Determine which edge building function to use
        # TODO Modify to fallback to `isinstance` where possible
        edge = None

        pre_c = list(c.pre.__class__.__mro__) + [None]
        post_c = list(c.post.__class__.__mro__) + [None]

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
    return edge


@register_build_edge(pre=nengo.Ensemble, post=nengo.Node)
def _ensemble_to_node(builder, c):
    # Get the vertices
    prevertex = builder.ensemble_vertices[c.pre]
    postvertex = builder.get_node_in_vertex(c)

    # Create the edge
    edge = edges.DecoderEdge(c, prevertex, postvertex)

    return edge


@register_build_edge(pre=nengo.Node, post=nengo.Ensemble)
def _node_to_ensemble(builder, c):
    # If the Node has constant output then add to the direct input
    # for the Ensemble and don't add an edge, otherwise add an
    # edge from the appropriate Rx element to the Ensemble.
    postvertex = builder.ensemble_vertices[c.post]
    if c.pre.output is not None and not callable(c.pre.output):
        postvertex.direct_input += np.dot(np.asarray(c.pre.output),
                                          np.asarray(c.transform).T)
    elif builder.config[c.pre].f_of_t:
        # Node is a function of time to be evaluated in advance
        if c.pre in builder.f_of_t_vertices:
            prevertex = builder.f_of_t_vertices[c.pre]
        else:
            prevertex = value_source_vertex.ValueSourceVertex(
                c.pre, builder.config[c.pre].f_period, builder.dt)
            builder.add_vertex(prevertex)
            builder.f_of_t_vertices[c.pre] = prevertex

        edge = edges.NengoEdge(c, prevertex, postvertex)
        return edge
    else:
        # Node is executed on host
        prevertex = builder.get_node_out_vertex(c)
        edge = edges.InputEdge(c, prevertex, postvertex,
                               filter_is_accumulatory=False)
        return edge


@register_build_edge(pre=nengo.Node, post=nengo.Node)
def _node_to_node(builder, c):
    builder.node_node_connections.append(c)
