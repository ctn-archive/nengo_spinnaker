import logging
import numpy as np
import sys
import time

import nengo
from pacman103.core import control

from . import builder
from . import nodes

logger = logging.getLogger(__name__)


class Simulator(object):
    """SpiNNaker simulator for Nengo models."""
    def __init__(self, model, machine_name=None, seed=None, io=None,
                 config=None):
        """Initialise the simulator with a model, machine and IO preferences.

        :param nengo.Network model: The model to simulate
        :param machine_name: Address of the SpiNNaker machine on which to
            simulate.  If `None` then the setting is taken out of the
            PACMAN configuration files.
        :type machine_name: string or None
        :param int seed: A seed for all random number generators used in
            building the model
        :param io: An IO interface builder from :py:mod:`nengo_spinnaker.io`
            or None.  The IO is used to allow host-based computation of Nodes
            to communicate with the SpiNNaker board. If None then an Ethernet
            connection is used by default.
        :param config: Configuration as required for components.
        """
        dt = 0.001

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

        (self.dao, host_network, self.probes) = \
            self.builder(model, dt, seed, node_builder=io, config=config)

        self.host_sim = None
        if not len(host_network.nodes) == 0:
            self.host_sim = nengo.Simulator(host_network, dt=dt)

        self.dao.writeTextSpecs = True

        self.dt = dt

    def run(self, time_in_seconds=None, clean=True):
        """Run the model for the specified amount of time.

        :param float time_in_seconds: The duration for which to simulate.
        :param bool clean: Remove all traces of the simulation from the board
            on completion of the simulation.  If False then you will need to
            execute an `app_stop` manually before running any later simulation.
        """
        self.time_in_seconds = time_in_seconds
        self.controller = control.Controller(sys.modules[__name__],
                                             self.machine_name)

        # Preparation functions, set the run time for each vertex
        for vertex in self.dao.vertices:
            vertex.runtime = time_in_seconds
            if hasattr(vertex, 'pre_prepare'):
                vertex.pre_prepare()

        # PACMANify!
        self.controller.dao = self.dao
        self.dao.set_hostname(self.machine_name)

        # TODO: Modify Transceiver so that we can manually check for
        # application termination  i.e., we want to do something during the
        # simulation time, not pause in the TxRx.
        self.dao.run_time = None

        self.controller.set_tag_output(1, 17895)  # Only required for Ethernet

        self.controller.map_model()

        # Preparation functions
        for vertex in self.dao.vertices:
            if hasattr(vertex, 'post_prepare'):
                vertex.post_prepare()

        self.controller.generate_output()
        self.controller.load_targets()
        self.controller.load_write_mem()

        # Start the IO and perform host computation
        with self.io as node_io:
            self.node_io = node_io
            self.controller.run(self.dao.app_id)
            node_io.start()

            current_time = 0.
            try:
                if self.host_sim is not None:
                    while (time_in_seconds is None or
                           current_time < time_in_seconds):
                        s = time.clock()
                        self.host_sim.step()
                        t = time.clock() - s

                        if t < self.dt:
                            time.sleep(self.dt - t)
                            t = self.dt
                        current_time += t
                else:
                    time.sleep(time_in_seconds)
            except KeyboardInterrupt:
                logger.debug("Stopping simulation.")

        # Retrieve any probed values
        logger.debug("Retrieving data from the board.")
        self.data = dict()
        for p in self.probes:
            self.data[p.probe] = p.get_data(self.controller.txrx)

        # Stop the application from executing
        logger.debug("Stopping the application from executing.")
        if clean:
            self.controller.txrx.app_calls.app_signal(self.dao.app_id, 2)

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        return dt * np.arange(int(self.time_in_seconds/dt))
