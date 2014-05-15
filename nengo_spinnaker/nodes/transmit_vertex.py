from ..utils import bins, filters, vertices


@filters.with_filters(2, 3)
class TransmitVertex(vertices.NengoVertex):
    """PACMAN Vertex for an object which receives input for a Node and
    transmits it to the host.
    """
    REGIONS = vertices.ordered_regions('SYSTEM')
    MODEL_NAME = "nengo_tx"

    def __init__(self, node, dt=0.001, output_period=100, time_step=1000,
                 constraints=None, label=None):
        self.node = node

        self.dt = dt
        self.time_step = time_step
        self.output_period = output_period

        self.filters = bins.FilterCollection()

        # Create the vertex
        super(TransmitVertex, self).__init__(
            1, constraints=constraints, label=label
        )

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        """Get the size (in words) of the SYSTEM region."""
        return 5

    @vertices.region_write('SYSTEM')
    def write_region_system(self, subvertex, spec):
        """Write the system region for the given subvertex."""
        # 1. Number of dimensions
        # 2. Machine time step in us
        # 3. Output period in ticks
        # 4. Number of filters
        # 5. Number of filter keys
        spec.write(data=self.node.size_in)
        spec.write(data=self.time_step)
        spec.write(data=self.output_period)

        if len(self.in_edges) > 0:
            spec.write(data=len(self.filters))
            spec.write(data=self.filters.num_keys(subvertex))
        else:
            spec.write(data=0, repeats=2)
