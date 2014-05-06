import collections
import logging
import numpy as np
import sys
import threading
import time

from nengo.utils.compat import is_callable

from pacman103.core import control
from pacman103 import conf

from . import builder
from . import node_builders

logger = logging.getLogger(__name__)


class Simulator(object):
    def __init__(self, model, dt=0.001, seed=None, io=None):
        # Build the model
        self.builder = builder.Builder()

        # Set up the IO
        self.machinename = conf.config.get('Machine', 'machineName')
        if io is None:
            io = node_builders.Ethernet(self.machinename)
        self.io = io

        (self.dao, self.nodes, self.node_node_connections) = self.builder(
            model, dt, seed, node_builder=io
        )
        self.dao.writeTextSpecs = True

        self.dt = dt

    def get_node_input(self, node):
        """Return the latest input for the given Node

        :return: an array of data for the Node, or None if no data received
        :raises KeyError: if the Node is not a valid Node
        """
        # Get the input from the board
        try:
            i = self.node_io.get_node_input(node)
        except KeyError:
            # Node does not receive input from the board
            i = np.zeros(node.size_in)

        if i is None or None in i:
            # Incomplete input, return None
            return None

        # Add Node->Node input if required
        if node not in self._internode_cache:
            return i

        i_s = self._internode_cache[node].values()

        if None in i_s:
            # Incomplete input, return None
            return None

        # Return input from board + input from other Nodes on host
        return np.sum([i, np.sum(i_s, axis=0)], axis=0)

    def set_node_output(self, node, output):
        """Set the output of the given Node

        :raises KeyError: if the Node is not a valid Node
        """
        # Output to board
        self.node_io.set_node_output(node, output)

        # Output to other Nodes on host
        if node in self._internode_out_maps:
            for (post, transform) in self._internode_out_maps[node]:
                self._internode_cache[post][node] = np.dot(transform, output)

    def run(self, time_in_seconds=None):
        """Run the model, currently ignores the time."""
        self.controller = control.Controller(
            sys.modules[__name__],
            conf.config.get('Machine', 'machineName')
        )

        # Preparation functions
        for vertex in self.dao.vertices:
            if hasattr(vertex, 'prepare_vertex'):
                vertex.prepare_vertex()

        # Create some caches for Node->Node connections, and a map of Nodes to
        # other Nodes on host
        # TODO: Filters on Node->Node connections
        self._internode_cache = collections.defaultdict(dict)
        self._internode_out_maps = collections.defaultdict(list)
        for c in self.node_node_connections:
            self._internode_cache[c.post][c.pre] = None
            self._internode_out_maps[c.pre].append((c.post, c.transform))

        # PACMANify!
        self.controller.dao = self.dao
        self.dao.set_hostname(self.machinename)
        self.dao.run_time = None  # TODO: Modify Transceiver so that we can
                                  # manually check for application termination
                                  # i.e., we want to do something during the
                                  # simulation time, not pause in the TxRx.
        self.controller.set_tag_output(1, 17895)  # Only required for Ethernet

        # TODO: All of the following will become more modular!
        self.controller.map_model()
        self.controller.generate_output()
        self.controller.load_targets()
        self.controller.load_write_mem()
        self.controller.run(self.dao.app_id)

        # Start the IO and perform host computation
        with self.io as node_io:
            self.node_io = node_io

            # Create the Node threads
            node_sims = [NodeSimulator(node, self, self.dt, time_in_seconds)
                         for node in self.nodes if is_callable(node.output)]

            # Sleep for simulation time/forever
            try:
                if time_in_seconds is not None:
                    time.sleep(time_in_seconds)
                else:
                    while True:
                        time.sleep(10.)
            except KeyboardInterrupt:
                pass
            finally:
                # Any necessary teardown functions
                for sim in node_sims:
                    sim.stop()


class NodeSimulator(object):
    """A "thread" to periodically evaluate a Node."""
    def __init__(self, node, simulator, dt, time_in_seconds):
        """Create a new NodeSimulator

        :param node: the `Node` to simulate
        :param simulator: the simulator, providing functions `get_node_input`
                          and `set_node_output`
        :param dt: timestep with which to evaluate the `Node`
        :param time_in_seconds: duration of the simulation
        """
        self.node = node
        self.simulator = simulator
        self.dt = dt
        self.time = time_in_seconds
        self.time_passed = 0.

        self.timer = threading.Timer(self.dt, self.tick)
        self.timer.start()

    def stop(self):
        """Permanently stop simulation of the Node."""
        self.time = 0.
        self.timer.cancel()

    def tick(self):
        """Simulate the Node and prepare the next timer tick if necessary."""
        start = time.clock()

        node_output = None

        if self.node.size_in > 0:
            node_input = self.simulator.get_node_input(self.node)

            if node_input is not None and None not in node_input:
                node_output = self.node.output(self.time_passed, node_input)
        else:
            node_output = self.node.output(self.time_passed)

        if node_output is not None:
            self.simulator.set_node_output(self.node, node_output)
        stop = time.clock()

        if stop - start > self.dt:
            self.dt = stop - start
            logger.warning("%s took longer than one timestep to simulate. "
                           "Decreasing frequency of evaluation." % self.node)

        self.time_passed += self.dt
        if self.time is None or self.time_passed < self.time:
            self.timer = threading.Timer(self.dt, self.tick)
            self.timer.start()
