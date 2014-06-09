import numpy as np

from ..utils import connections, fp, vertices


class ValueSourceVertex(vertices.NengoVertex):
    """ValueSource vertices contain the precomputed output of Nodes which are
    functions of time and play these values back into the simulation.
    """
    REGIONS = vertices.ordered_regions('SYSTEM', 'OUTPUT_KEYS', 'DATA')
    MODEL_NAME = "nengo_value_source"

    def __init__(self, node, node_period=0, dt=0.001, time_step=1000,
                 constraints=None, label=None):
        super(ValueSourceVertex, self).__init__(
            1, constraints=constraints, label="ValueSource %s" % node
        )
        self.node = node
        self.node_period = node_period
        self.time_step = time_step
        self.dt = dt

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    def dtcm_usage(self, n_atoms):
        # 40KB block swap space + some stuff
        return 10*1024

    def pre_prepare(self):
        # Compute the number of samples which will be stored
        if self.node_period is not None:
            self._periodic = True
            self._n_ticks = int(self.node_period / self.dt)
        else:
            if self.runtime is None:
                raise Exception("Cannot run an aperiodic signal for an "
                                "unbounded duration.  Consider setting a "
                                "period or removing the function of time "
                                "directive.")
            self._periodic = False
            self._n_ticks = int(self.runtime / self.dt)

        # Data is split into blocks of 20KB, though a block may be less than
        # this.  Each block is pulled into DTCM sequentially.
        self.connections = connections.Connections([e.conn for e in self.out_edges])
        self.width = self.connections.width
        self.data_size = self._n_ticks * self.width
        self.frames_per_block = 5*1024 / self.width  # 20KB / 4*t
        self.full_blocks = self._n_ticks / self.frames_per_block
        self.r_blocks = self._n_ticks % self.frames_per_block

        ts = np.arange(0, self._n_ticks * self.dt, self.dt)  # Eval points

        data = []
        for t in ts:
            v = np.array(self.node.output(t))

            output = []
            for tf in self.connections.transforms_functions:
                output.append(
                    np.dot(tf.transform, v if tf.function is None else
                           tf.function(v)))

            data.append(np.vstack(output))
        data = np.array(data)
        self.data = data.reshape((1, data.size))

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_system(self, n_atoms):
        # Time step
        # Number of output keys/dimensions
        # Periodic?
        # Length of full block (in frames)
        # Number of full blocks
        # Length of last block (in frames)
        return 6

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_keys(self, n_atoms):
        return self.width

    @vertices.region_pre_sizeof('DATA')
    def sizeof_data(self, n_atoms):
        return self.data_size

    @vertices.region_write('SYSTEM')
    def write_system(self, subvertex, spec):
        spec.write(data=self.time_step)
        spec.write(data=self.width)
        spec.write(data=0x1 if self._periodic else 0x0)
        spec.write(data=self.full_blocks)
        spec.write(data=self.frames_per_block)
        spec.write(data=self.r_blocks)

    @vertices.region_write('OUTPUT_KEYS')
    def write_keys(self, subvertex, spec):
        (x, y, p) = subvertex.placement.processor.get_coordinates()

        for (i, t) in enumerate(self.connections.transforms_functions):
            for d in range(t.transform.shape[0]):
                spec.write(data=(x << 24) | (y << 16) | ((p-1) << 11) |
                                (i << 6) | d)

    @vertices.region_write('DATA')
    def write_data(self, subvertex, spec):
        spec.write_array(data=fp.bitsk(self.data))

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.connections[subedge.edge.conn]
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key, 0xFFFFFFC0
