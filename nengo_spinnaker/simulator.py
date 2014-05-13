import collections
import logging
import numpy as np
import sys
import threading
import time

from nengo.utils.compat import is_callable

from pacman103.core import control
from pacman103.lib import lib_map

from . import builder
from . import nodes

logger = logging.getLogger(__name__)


class Simulator(object):
    def __init__(self, model, machine_name=None, dt=0.001, seed=None, io=None):
        # Get the hostname
        if machine_name is None:
            import ConfigParser
            from pacman103 import conf
            try:
                machine_name = conf.config.get("Machine", "machineName")
            except ConfigParser.Error:
                machine_name = None

            if machine_name is None or machine_name == "None":
                raise Exception("You must specify a SpiNNaker machine as "
                                "either an option to the Simulator or in a "
                                "PACMAN103 configuration file.")

        self.machine_name = machine_name

        # Set up the IO
        if io is None:
            io = nodes.Ethernet(self.machine_name)
        self.io = io

        # Build the model
        self.builder = builder.Builder()

        (self.dao, self.nodes, self.node_node_connections) = self.builder(
            model, dt, seed, node_builder=io
        )
        self.dao.writeTextSpecs = True

        self.dt = dt

    def get_node_input(self, node):
        """Return the latest input for the given Node

        :return: None if the input data is not complete, otherwise a tuple
                 containing filtered input from the board and a dict of input
                 from different Nodes.
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
            return i, {}

        with self._internode_cache_lock:
            i_s = self._internode_cache[node]

        if None in i_s.values():
            print "Incomplete Node->Node input", node, i_s
            # Incomplete input, return None
            return None

        # Return input from board, input from other Nodes on host
        return i, i_s

    def set_node_output(self, node, output):
        """Set the output of the given Node

        :raises KeyError: if the Node is not a valid Node
        """
        # Output to board
        if is_callable(node.output):
            self.node_io.set_node_output(node, output)

        # Output to other Nodes on host
        if node in self._internode_out_maps:
            with self._internode_cache_lock:
                for (post, transform) in self._internode_out_maps[node]:
                    self._internode_cache[post][node] = np.dot(transform,
                                                               output)

    def run(self, time_in_seconds=None, clean=True):
        """Run the model for the specified amount of time."""
        self.controller = control.Controller(sys.modules[__name__],
                                             self.machine_name)
        # Preparation functions
        for vertex in self.dao.vertices:
            if hasattr(vertex, 'prepare_vertex'):
                vertex.prepare_vertex()

        # Create some caches for Node->Node connections, and a map of Nodes to
        # other Nodes on host
        self._internode_cache = collections.defaultdict(dict)
        self._internode_out_maps = collections.defaultdict(list)
        self._internode_filters = collections.defaultdict(dict)
        for c in self.node_node_connections:
            self._internode_cache[c.post][c.pre] = None
            self._internode_out_maps[c.pre].append((c.post, c.transform))

            ftc = np.exp(-self.dt/c.synapse)
            self._internode_filters[c.post][c.pre] = (ftc, 1. - ftc)

        self._internode_cache_lock = threading.Lock()

        # PACMANify!
        self.controller.dao = self.dao
        self.dao.set_hostname(self.machine_name)
        self.dao.run_time = None  # TODO: Modify Transceiver so that we can
                                  # manually check for application termination
                                  # i.e., we want to do something during the
                                  # simulation time, not pause in the TxRx.
        self.controller.set_tag_output(1, 17895)  # Only required for Ethernet

        # TODO: All of the following will become more modular!
        self.controller.map_model()
        self.controller.generate_output()

        # Write the runtime to each used core, UINT32_MAX means "run forever"
        run_ticks = ((1 << 32) - 1 if time_in_seconds is None else
                     time_in_seconds * 1000)  # TODO Deal with timestep scaling
        for vertex in self.dao.vertices:
            if not vertex.virtual:
                for subvertex in vertex.subvertices:
                    x, y, p = subvertex.placement.processor.get_coordinates()
                    addr = 0xe5007000 + 128 * p + 116  # Space reserved for _p_
                    self.dao.mem_write_targets.append(lib_map.MemWriteTarget(
                        x, y, p, addr, run_ticks
                    ))

        self.controller.load_targets()
        self.controller.load_write_mem()
        self.controller.run(self.dao.app_id)

        # Start the IO and perform host computation
        with self.io as node_io:
            self.node_io = node_io

            # Create the Node threads
            for node in self.nodes:
                if not is_callable(node.output):
                    self.set_node_output(node, node.output)

            node_sims = [NodeSimulator(node, self, self.dt, time_in_seconds,
                                       self._internode_filters[node])
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

            # Stop the application from executing
            if clean:
                self.controller.txrx.app_calls.app_signal(self.dao.app_id, 2)


class NodeSimulator(object):
    """A "thread" to periodically evaluate a Node."""
    def __init__(self, node, simulator, dt, time_in_seconds, infilters={}):
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
        self.infilters = infilters

        self.filtered_inputs = collections.defaultdict(lambda : 0.)

        self.timer = threading.Timer(self.dt, self.tick)
        self.timer.name = "%sEvalThread" % self.node
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

            if node_input is not None:
                # Filter the inputs
                for (node, value) in node_input[1].items():
                    self.filtered_inputs[node] = (
                        value * self.infilters[node][0] +
                        self.filtered_inputs[node] * self.infilters[node][1]
                    )

                # Sum the inputs
                complete_input = (np.sum(self.filtered_inputs.values()) +
                                  node_input[0])
                node_output = self.node.output(self.time_passed,
                                               complete_input)
        else:
            node_output = self.node.output(self.time_passed)

        if node_output is not None:
            self.simulator.set_node_output(self.node, node_output)
        stop = time.clock()

        if stop - start > self.dt:
            self.dt = stop - start
            logger.warning("%s took longer than one timestep to simulate. "
                           "Decreasing frequency of evaluation." % self.node)

        self.time_passed += self.dt + (stop - start)
        if self.time is None or self.time_passed < self.time:
            self.timer = threading.Timer(self.dt, self.tick)
            self.timer.name = "EvalThread(%s)" % self.node
            self.timer.start()
