import json
import logging
from six import itervalues

from .assembler import Assembler
from .io.ethernet import EthernetIO
from .simulator import Simulator
from .spinnaker import partitioners
from .spinnaker.vertices import (dtcm_partitioner_constraint,
                                 sdram_partitioner_constraint,
                                 atoms_partitioner_constraint)

logger = logging.getLogger(__name__)


class Analyser(Simulator):
    """Static analysis tool for Nengo simulations to be run on SpiNNaker."""
    def __init__(self, network, dt=0.001, time_scaling=1.0, config=None,
                 io_type=EthernetIO):
        # Run through all the preparation steps that will be performed by the
        # Simulator.
        super(Analyser, self).__init__(network, dt=dt,
                                       time_scaling=time_scaling,
                                       config=config, io_type=io_type)

        # Assemble the model. TODO remove this step once it is moved into the
        # Simulator constructor.
        # NOTE runtime is 0!
        logger.info("Assembling model.")
        (self.vertices, self.edges) = Assembler.assemble(
            self.model, self.config, self.rngs, 0.0, self.dt,
            self.machine_timestep
        )

        # Partition the model.
        partitions = partitioners.partition_vertices(
            self.vertices, [dtcm_partitioner_constraint,  # Limit DTCM usage
                            sdram_partitioner_constraint,  # SDRAM
                            atoms_partitioner_constraint,  # Limit atoms
                            ]
        )
        self.split_vertices = partitioners.get_split_vertices(partitions)
        self.split_edges = partitioners.get_split_edges(self.edges,
                                                        self.split_vertices)

    def create_hypergraph_as_json(self, filename):
        """Write out the hypergraph describing network connectivity as JSON
        formatted data to a file.
        """
        # TODO Include (hyper)edge weights.
        # Create the data structure
        hypergraph = list()
        for hyperedge in itervalues(self.split_edges):
            nodes = (set(e.pre_split for e in hyperedge) |
                     set(e.post_split for e in hyperedge))
            hypergraph.append([repr(n) for n in nodes])

        # Dump to file
        with open(filename, 'w+') as f:
            json.dump(hypergraph, f)
