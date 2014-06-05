import logging
logger = logging.getLogger(__name__)

import utils
from utils import filters, vertices


@filters.with_filters(2, 3)
class ValueSinkVertex(vertices.NengoVertex):
    """ValueSinkVertex records the decoded values which it receives.
    """
    REGIONS = vertices.ordered_regions('SYSTEM', **{'VALUES': 15})
    MODEL_NAME = "nengo_value_sink"

    def __init__(self, width, dt=0.001, timestep=1000):
        super(ValueSinkVertex, self).__init__(1, constraints=None,
                                              label="Nengo Value Sink")
        self.width = width
        self.timestep = timestep
        self.dt = dt

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    def dtcm_usage(self, n_atoms):
        return self.sizeof_system(n_atoms) * 4

    @vertices.region_pre_prepare('SYSTEM')
    def prepare_system(self):
        # Calculate the number of ticks of execution
        self.run_ticks = ((1 << 32) - 1 if self.runtime is None else
                          self.runtime * 1000)

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_system(self, n_atoms):
        # 1. Timestep
        # 2. Width
        return 2

    @vertices.region_pre_sizeof('VALUES')
    def sizeof_values(self, n_atoms):
        if self.runtime is None:
            logger.warning("Can't record for an indefinite period, "
                           "not recording at all.")
            return 0
        else:
            return self.width * self.run_ticks

    @vertices.region_write('SYSTEM')
    def write_system(self, subvertex, spec):
        spec.write(data=self.timestep)
        spec.write(data=self.width)
