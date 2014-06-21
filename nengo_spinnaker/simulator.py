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
    """SpiNNaker simulator for Nengo models.

    In general Probes return data in the same form as Nengo and data can be
    accessed using the :py:attr:`data` dictionary.

    ::

         output = sim.data[probe]

    Spike probes
        The one current exception to this rule is spike data.  Spike data from
        Ensembles is formatted as a list of spike times for each neuron.  This
        allows it to be used directly with
        :py:func:`matplotlib.pyplot.eventplot`::

            model = nengo.Network()
            with model:
                target = nengo.Ensemble(25, 1)
                p = nengo.Probe(target, 'spikes')

            sim = nengo_spinnaker.Simulator(model)
            sim.run(10.)

            plt.eventplot(sim.data[p], colors=[[0, 0, 1]])

    Voltage probes
        Currently not supported.

    :attr data: A dictionary mapping Probes to the data they probed.
    """
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
        self.executed = False

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
        if len(host_network.nodes) > 0:
            self.host_sim = nengo.Simulator(host_network, dt=dt)

        self.dt = dt

    def _prepare_model_for_execution(self, time_in_seconds):
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

    def run(self, time_in_seconds=None, clean=True):
        """Run the model for the specified amount of time.

        .. warning::

            Unlike the reference simulator you may not run the Nengo/SpiNNaker
            simulator more than once without performing a reset.
            For example, the second invocation of :py:func:`run` will raise a
            :py:exc:`NotImplementedError`::

                sim.run(1.)
                # Do some stuff
                sim.run(1.)  # Raises NotImplementedError

            If you do want to run the Simulator twice you will have to call
            :py:func:`~nengo_spinnaker.Simulator.reset`, but this will restart
            the simulation from the beginning::

                sim.run(1.)
                # Do some stuff
                sim.reset()
                sim.run(1.)  # Will start from t=0

        :param float time_in_seconds: The duration for which to simulate.
        :param bool clean: Remove all traces of the simulation from the board
            on completion of the simulation.  If False then you will need to
            execute an `app_stop` manually before running any later simulation.
        """
        if self.executed:
            # This is to stop the use case of:
            # >>> sim.run(10)
            # >>> # Do stuff..
            # >>> sim.run(10)
            # which we currently can't support.  We've implemented the `reset`
            # function so that scripts can match between reference and
            # SpiNNaker provided they don't require multiple re-runs.
            raise NotImplementedError(
                "You must reset before running this Simulator again.")

        self.time_in_seconds = time_in_seconds
        self.controller = control.Controller(sys.modules[__name__],
                                             self.machine_name)

        try:
            # Run the PACMAN place/partition, route, data spec tools
            self.executed = True
            self._prepare_model_for_execution(time_in_seconds)

            # Start the IO and perform host computation
            with self.io as node_io:
                self.controller.run(self.dao.app_id)
                node_io.start()

                current_time = 0.
                try:
                    if self.host_sim is not None:
                        while (time_in_seconds is None or
                               current_time < time_in_seconds):
                            # Execute a single step of the host simulator and
                            # measure how long it takes.
                            s = time.clock()
                            self.host_sim.step()
                            t = time.clock() - s

                            # If it takes less than one time step then sleep
                            # for the remaining time
                            if t < self.dt:
                                time.sleep(self.dt - t)
                                t = self.dt

                            # TODO: Currently if one step of the simulator
                            # takes more than one time step we can't do
                            # anything, so the host lags behind the board.
                            # We should request that we can modify the time
                            # step of the reference simulator to stretch the
                            # time steps on the host so that it stays in
                            # step with the board, albeit at a lower sample
                            # rate.
                            #
                            # if t > dt:
                            #     self.host_sim.dt = t

                            # Keep track of how long we've been running for
                            current_time += t
                    else:
                        # If there are no Nodes to simulate on the host then we
                        # either sleep for the specified run time, or we sleep
                        # in increments of 10s.
                        if time_in_seconds is not None:
                            time.sleep(time_in_seconds)
                        else:
                            while True:
                                time.sleep(10.)
                except KeyboardInterrupt:
                    logger.debug("Stopping simulation.")

            # Retrieve any probed values
            logger.debug("Retrieving data from the board.")
            self.data = dict()
            for p in self.probes:
                self.data[p.probe] = p.get_data(self.controller.txrx)

        finally:
            # Stop the application from executing
            try:
                logger.debug("Stopping the application from executing.")
                if clean:
                    # TODO: At some point this will become a clearer call to
                    # SpiNNaker manager library, at the moment this just says
                    # "Send signal 2 (meaning stop) to all executables with the
                    #  app_id we've given them (usually 30)."
                    self.controller.txrx.app_calls.app_signal(
                        self.dao.app_id, 2)
            except Exception:
                pass

    def reset(self):
        """Reset the Simulator.

        The next simulation will start from the beginning.
        """
        # This is only really here to ensure that the behaviour is consistent
        # with the reference simulator.  We currently don't allow multiple runs
        # in sequence, so there must be a reset capability to allow reuse of
        # the simulator.
        self.executed = False

    def trange(self, dt=None):
        """Generate a list of time steps for the last simulation.

        :returns: Numpy array of time steps.
        """
        if self.time_in_seconds is not None:
            dt = self.dt if dt is None else dt
            return dt * np.arange(int(self.time_in_seconds/dt))
        else:
            # XXX: This is horrible, but it beats getting an exception because
            #      dividing None is invalid.  If the run time wasn't specified
            #      probing would have been disabled.
            # TODO: Allow some probing for unspecified run time... Will require
            #       writing the final run time back somehow. (When we have
            #       masses of bandwidth we could even probe on
            #       on host)
            #       The other option is to dynamically switch the model so that
            #       we probe as best as we can on host when no run time is
            #       specified.  This would require some rearranging of the
            #       Builder.
            return np.array([])
