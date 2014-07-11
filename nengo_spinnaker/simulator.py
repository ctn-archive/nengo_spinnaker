import logging
import numpy as np
import sys
import time
import platform

import nengo
from pacman103.core import control

from . import assembler
from . import builder
from . import nodes
from .config import Config
import utils

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
        self.dt = dt
        self.executed = False
        self.config = config if config is not None else Config()

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
        (self.objs, self.conns, self.keyspace) =\
            builder.Builder.build(model, dt, seed)

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

        # Swap out function of time nodes
        objs = list()
        conns = list()

        replaced_nodes = dict()
        for obj in self.objs:
            if isinstance(obj, nengo.Node):
                if self.config[obj].f_of_t:
                    # Get the likely size of this object
                    out_conns = utils.connections.Connections(
                        [c for c in self.conns if c.pre == obj])
                    width = out_conns.width

                    # Get the overall duration of the signal
                    p_durations = [t for t in [time_in_seconds,
                                               self.config[obj].f_period] if
                                   t is not None]

                    if len(p_durations) == 0:
                        # Indefinite simulation with indefinite function, will
                        # have to simulate on host.
                        self.config[obj].f_of_t = False
                        objs.append(obj)
                        continue

                    duration = min(p_durations)
                    periodic = (self.config[obj].f_period is not None and
                                self.config[obj].f_period == duration)

                    if width * duration > 6 * 1024**2:
                        # Storing this function (and all its transforms) would
                        # take up too much memory, will have to simulate on
                        # host.
                        # TODO Split up the connections to reduce the memory
                        #      usage instead of giving up.
                        self.config[obj].f_of_t = False
                        objs.append(obj)
                        continue

                    # It is possible to fit the function (and all its
                    # transforms) in memory, so replace it with a
                    # function of time vertex.
                    new_obj = assembler.ValueSource.from_node(
                        obj.output, out_conns, duration, periodic, self.dt)
                    replaced_nodes[obj] = new_obj
                    objs.append(new_obj)
                else:
                    objs.append(obj)
            else:
                objs.append(obj)

        for c in self.conns:
            if c.pre in replaced_nodes:
                c.pre = replaced_nodes[c.pre]
            conns.append(c)

        # Set up the host network for simulation
        host_network = utils.nodes.create_host_network(
            [c.to_connection() if isinstance(c.pre, nengo.Node) and
             isinstance(c.post, nengo.Node) else c for c in conns],
            self.io, self.config)

        # Prepare the network for IO
        (objs, conns) = self.io.prepare_network(objs, conns, self.dt,
                                                self.keyspace)

        # Assemble the model for simulation
        asmblr = assembler.Assembler()
        vertices, edges = asmblr(
            objs, conns, time_in_seconds, self.dt)

        # Set up host simulator
        host_sim = nengo.Simulator(host_network, dt=self.dt)

        # Build the list of probes
        self.probes = list()
        for vertex in vertices:
            if isinstance(vertex, assembler.DecodedValueProbe):
                self.probes.append(
                    utils.probes.DecodedValueProbe(vertex, vertex.probe))
            else:
                if hasattr(vertex, 'probes'):
                    for probe in vertex.probes:
                        if probe.attr == 'spikes':
                            self.probes.append(
                                utils.probes.SpikeProbe(vertex, probe))

        # PACMANify!
        for vertex in vertices:
            self.controller.add_vertex(vertex)

        for edge in edges:
            self.controller.add_edge(edge)

        # TODO: Modify Transceiver so that we can manually check for
        # application termination  i.e., we want to do something during the
        # simulation time, not pause in the TxRx.
        self.controller.dao.run_time = None

        self.controller.set_tag_output(1, 17895)  # Only reqd. for Ethernet
        self.controller.map_model()
        self.controller.generate_output()

        try:
            self.controller.load_targets()
            self.controller.load_write_mem()

            # Start the IO and perform host computation
            with self.io as node_io:
                self.controller.run(self.controller.dao.app_id)
                node_io.start()

                current_time = 0.
                try:
                    if host_sim is not None:
                        if platform.system() == 'Windows':
                            while (time_in_seconds is None or
                                   current_time < time_in_seconds):
                                # Execute a single step of the host simulator
                                # and measure how long it takes.
                                s = time.clock()
                                host_sim.step()
                                t = time.clock() - s

                                # If it takes less than one time step then
                                # sleep for the remaining time
                                if t < host_sim.dt:
                                    t = time.clock() - s
                                    # Note: time.sleep() sucks on windows
                                    # time.sleep(host_sim.dt - t)

                                current_time += t
                        else:
                            while (time_in_seconds is None or
                                   current_time < time_in_seconds):
                                # Execute a single step of the host simulator
                                # and measure how long it takes.
                                s = time.clock()
                                host_sim.step()
                                t = time.clock() - s

                                # If it takes less than one time step then
                                # sleep for the remaining time
                                if t < host_sim.dt:
                                    time.sleep(host_sim.dt - t)
                                    t = host_sim.dt

                                # TODO: Currently if one step of the simulator
                                # takes more than one time step we can't do
                                # anything, so the host lags behind the board.
                                # We should request that we can modify the time
                                # step of the reference simulator to stretch
                                # the time steps on the host so that it stays
                                # in step with the board, albeit at a lower
                                # sample rate.
                                #
                                # if t > host_sim.dt:
                                #     host_sim.dt = t

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
            if time_in_seconds is not None:
                for p in self.probes:
                    self.data[p.probe] = p.get_data(self.controller.txrx)

        finally:
            # Stop the application from executing
            try:
                logger.info("Stopping the application from executing.")
                if clean:
                    # TODO: At some point this will become a clearer call to
                    # SpiNNaker manager library, at the moment this just says
                    # "Send signal 2 (meaning stop) to all executables with the
                    #  app_id we've given them (usually 30)."
                    time.sleep(0.1)
                    self.controller.txrx.app_calls.app_signal(
                        self.controller.dao.app_id, 2)
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
            # TODO: Allow some probing for unspecified run time... Will require
            #       writing the final run time back somehow. (When we have
            #       masses of bandwidth we could even probe on
            #       on host)
            #       The other option is to dynamically switch the model so that
            #       we probe as best as we can on host when no run time is
            #       specified.  This would require some rearranging of the
            #       Builder.
            raise NotImplementedError('Cannot provide time steps for '
                                      'indefinite run time.')
