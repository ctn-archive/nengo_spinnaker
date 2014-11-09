import logging

from pacman.model.partitionable_graph.partitionable_graph import \
    PartitionableGraph

from .assembler import Assembler
from .builder import Builder
from .config import Config
from .io.ethernet import EthernetIO


logger = logging.getLogger(__name__)


class Simulator(object):
    """SpiNNaker Simulator for Nengo models."""

    def __init__(self, network, dt=0.001, time_scaling=1.0, config=None,
                 io_type=EthernetIO):
        """Initialise the simulator with a Nengo network to simulate.

        Parameters
        ----------
        network : :py:class:`nengo.Network`
            A network object to be built and then simulated.
        dt : float
            The length of a simulation step in seconds.
        time_scaling : float
            The ratio of simulation execution time to simulated time, default
            value of 1.0 results in simulation in real time.
        config : :py:class:`~.config.Config` or None
            Specific Nengo/SpiNNaker configuration options.
        io_type : {EthernetIO}
            A type which can perform the tasks necessary to communicate with a
            running simulation (including preparing the model for this
            communication).
        """
        # Save the config object, or create a empty one if `None`
        self.config = config
        if self.config is None:
            self.config = Config()

        # Save dt, compute machine timestep (in usec)
        self.dt = dt
        self.machine_timestep = int(self.dt * time_scaling * 10**6)

        # Use the Builder system to build the network into a connection tree
        logger.info("Building model into intermediate representation.")
        model, self.rngs = Builder.build(network, self.config)

        # Process the connection tree for IO handling
        logger.info("Preparing model IO for simulation.")
        self.model = io_type.prepare_connection_tree(model)
        self.network = network

    def run(self, time_in_seconds=None):
        """Simulate for the given period of time, if no period of time is given
        then the model will be simulated indefinitely.

        .. todo::
            The aim is to move most of the SpiNNaker prepare and load function
            into the :py:func:`~Simulator.__init__` method.  The result will be
            that pressing :py:func:`run` is sufficient to just start the model
            simulating for a period of time.  Moving to this new model will
            require:

             - Retrieving blocks of probe data during simulation.
             - Loading function of time data during simulation.
             - Precise control of simulation.

            Basically, steps 1-7 below can move above the crease.

        .. warning::
            The nature of the simulation process means that it is not possible
            to perform sequential simulations of a model without reloading the
            model to SpiNNaker, which will restart the simulation from time 0.
            To maintain API equivalence with Nengo it is necessary to call
            :py:func:`~Simulator.reset` between calls to `run`.  **It is not
            intended that this state of affairs continue.**

        Parameters
        ----------
        time_in_seconds : float or None
            The simulation time in seconds, or None to signify that the
            simulation should run indefinitely.

        """
        # Assemble the model for simulation
        logger.info("Run step 1/9: Assembling model for simulation.")
        (vertices, edges) = Assembler.assemble(
            self.model, self.config, self.rngs, time_in_seconds, self.dt,
            self.machine_timestep
        )

        # Convert the model into a graph for mapping to SpiNNaker
        graph = PartitionableGraph(label=self.network.label, vertices=vertices,
                                   edges=edges)

        # Partition and place
        logger.info("Run step 2/9: Partitioning and placing the model.")

        # Route edges from the model
        logger.info("Run step 3/9: Routing connections on the SpiNNaker "
                    "network.")

        # Generate the data to be loaded to SpiNNaker
        logger.info("Run step 4/9: Generating data to load to SpiNNaker.")

        # Load the data to SpiNNaker
        logger.info("Run step 5/9: Loading data to SpiNNaker.")

        # Load the executables to SpiNNaker
        logger.info("Run step 6/9: Loading executables to SpiNNaker.")

        # Generate the network to simulate on host
        logger.info("Run step 7/9: Generating the portion of the model to "
                    "simulate on this computer.")

        # Start the simulation
        logger.info(
            "Run step 8/9: Starting the simulation (run for {})."
            .format("ever" if time_in_seconds is None else "{:3f}s"
                .format(time_in_seconds))
        )

        # Stop the simulation and retrieve data from SpiNNaker
        logger.info("Run step 9/9: Stopping the simulation and retrieving "
                    "probed values.")
