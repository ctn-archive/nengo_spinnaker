import logging

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
        model = Builder.build(network, self.config)

        # Process the connection tree for IO handling
        logger.info("Preparing model IO for simulation.")
        model = io_type.prepare_connection_tree(model)
