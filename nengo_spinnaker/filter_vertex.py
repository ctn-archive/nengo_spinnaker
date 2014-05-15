from .utils import bins, filters, vertices


@filters.with_filters(3, 4)
class FilterVertex(vertices.NengoVertex):
    """PACMAN Vertex for a filtered external output node."""
    REGIONS = vertices.ordered_regions('SYSTEM', 'OUTPUT_KEYS')
    MODEL_NAME = "nengo_filter"

    def __init__(self, dimensions, output_id, dt=0.001, time_step=1000,
                 output_period=100, constraints=None, label='filter'):
        """Create a new FilterVertex

        :param dimensions: number of values
        :param output_id: id key to place in packet routing
        :param time_step: Machine timestep (in microseconds)
        :param output_period: Time between output events (in ticks)
        """
        self.time_step = time_step
        self.output_id = output_id
        self.output_period = output_period
        self.dt = dt

        self.dimensions = dimensions

        self.filters = bins.FilterCollection()

        # Create the vertex
        super(FilterVertex, self).__init__(1,
                                           constraints=constraints,
                                           label=label)

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        """Get the size (in words) of the SYSTEM region."""
        return 5

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_region_output_keys(self, n_atoms):
        """Get the size (in words) of the OUTPUT_KEYS region."""
        return self.dimensions

    def cpu_usage(self, lo_atom, hi_atom):
        """Return the CPU utilisation for the specified atoms."""
        # TODO: Calculate this
        return 0

    def get_maximum_atoms_per_core(self):
        return 1

    @vertices.region_write('SYSTEM')
    def write_region_system(self, subvertex, spec):
        """Write the system region for the given subvertex."""
        # 1. Number of dimensions
        # 2. Machine time step in us
        # 3. Output period in ticks
        # 4. Number of filters
        # 5. Number of filter keys
        spec.write(data=self.dimensions)
        spec.write(data=self.time_step)
        spec.write(data=self.output_period)

        if len(self.in_edges) > 0:
            spec.write(data=len(self.filters))
            spec.write(data=self.filters.num_keys(subvertex))
        else:
            spec.write(data=0, repeats=2)

    @vertices.region_write('OUTPUT_KEYS')
    def write_region_output_keys(self, subvertex, spec):
        """Write the output keys region for the given subvertex."""
        x, y, p = subvertex.placement.processor.get_coordinates()
        i = self.output_id
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        for d in range(self.dimensions):
            spec.write(data=key | d)

    def generate_routing_info(self, subedge):
        """Generate a key and mask for the given subedge."""
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        i = self.output_id
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

        return key, 0xFFFFFFE0
