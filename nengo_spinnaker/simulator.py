import sys
import logging
import threading
import time

from nengo.utils.compat import is_callable

from pacman103.core import control
from pacman103 import conf
from pacman103.core.spinnman.interfaces import iptag

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

        self.dao = self.builder(model, dt, seed, node_builder=io)
        self.dao.writeTextSpecs = True

        self.dt = dt

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
            # Create the Node threads
            node_sims = [NodeSimulator(node, node_io, self.dt, time_in_seconds)
                         for node in self.io.nodes if is_callable(node.output)]

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
    def __init__(self, node, io, dt, time_in_seconds):
        """Create a new NodeSimulator

        :param node: the `Node` to simulate
        :param io: the IO handler, providing functions `get_node_input` and
                   `set_node_output`
        :param dt: timestep with which to evaluate the `Node`
        :param time_in_seconds: duration of the simulation
        """
        self.node = node
        self.io = io
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
            node_input = self.io.get_node_input(self.node)

            if node_input is not None and None not in node_input:
                node_output = self.node.output(self.time_passed, node_input)
        else:
            node_output = self.node.output(self.time_passed)

            if node_output is not None:
                self.io.set_node_output(self.node, node_output)
        stop = time.clock()

        if stop - start > self.dt:
            self.dt = stop - start
            logger.warning("%s took longer than one timestep to simulate. "
                           "Decreasing frequency of evaluation." % self.node)

        self.time_passed += self.dt
        if self.time is None or self.time_passed < self.time:
            self.timer = threading.Timer(self.dt, self.tick)
            self.timer.start()
