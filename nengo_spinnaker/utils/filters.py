from . import fixpoint as fp
from vertices import region_pre_sizeof, region_sizeof, region_write


def with_filters(filter_id=14, routing_id=15):
    """Add input filtering to the given NengoVertex subclass.

    :param filter_id: region ID to use for filters
    :param routing_id: region ID to use for filter routing entries
    """
    def cls_(cls):
        cls.REGIONS.update({"FILTERS": filter_id,
                            "FILTER_ROUTING": routing_id})
        cls._sizeof_region_filters = _sizeof_region_filters
        cls._pre_sizeof_region_filter_routing = \
            _pre_sizeof_region_filter_routing
        cls._sizeof_region_filter_routing = _sizeof_region_filter_routing

        cls._write_region_filters = _write_region_filters
        cls._write_region_filter_routing = _write_region_filter_routing
        return cls
    return cls_


@region_pre_sizeof("FILTERS")
def _sizeof_region_filters(self, n_atoms):
    # 3 words per filter + 1 for length
    return 3 * len(self.filters) + 1


@region_pre_sizeof("FILTER_ROUTING")
def _pre_sizeof_region_filter_routing(self, n_atoms):
    return 3 * len(self.in_edges) * 5


@region_sizeof("FILTER_ROUTING")
def _sizeof_region_filter_routing(self, subvertex):
    # 3 words per entry, 1 entry per in_subedge + 1 for length
    return 3 * self.filters.num_keys(subvertex) + 1


@region_write("FILTERS")
def _write_region_filters(self, subvertex, spec):
    # spec.write(data=len(self.filters))
    for f_ in self.filters:
        f = f_.get_filter_tc(self.dt)
        spec.write(data=fp.bitsk(f[0]))
        spec.write(data=fp.bitsk(f[1]))
        spec.write(data=f_.accumulator_mask)


@region_write("FILTER_ROUTING")
def _write_region_filter_routing(self, subvertex, spec):
    # For each incoming subedge we write the key, mask and index of the
    # filter to which it is connected.  At some later point we can try
    # to combine keys and masks to minimise the number of comparisons
    # which are made in the SpiNNaker application.
    # spec.write(data=self.filters.num_keys(subvertex))
    for kms in self.filters.get_indexed_keys_masks(subvertex):
        for km in kms:
            spec.write(data=km[0])
            spec.write(data=km[1])
            spec.write(data=km[2])


class VertexWithFilters(object):
    def sizeof_region_filters(self):
        # 3 words per filter
        return 4 * 3 * len(self.filters)

    def sizeof_region_filter_keys(self, subvertex):
        # 3 words per entry
        # 1 entry per in_subedge
        return 4 * 3 * self.filters.num_keys(subvertex)

    def write_region_filters(self, subvertex):
        """Write the filter parameters."""
        subvertex.spec.switchWriteFocus(self.REGIONS.FILTERS)
        subvertex.spec.comment("# Filter Parameters")
        for f_ in self.filters:
            f = f_.get_filter_tc(self.dt)
            subvertex.spec.write(data=fp.bitsk(f[0]))
            subvertex.spec.write(data=fp.bitsk(f[1]))
            subvertex.spec.write(data=f_.accumulator_mask)

    def write_region_filter_keys(self, subvertex):
        # Write the filter routing entries
        subvertex.spec.switchWriteFocus(self.REGIONS.FILTER_ROUTING)
        subvertex.spec.comment("# Filter Routing Keys and Masks")
        """
        For each incoming subedge we write the key, mask and index of the
        filter to which it is connected.  At some later point we can try
        to combine keys and masks to minimise the number of comparisons
        which are made in the SpiNNaker application.
        """
        for kms in self.filters.get_indexed_keys_masks(subvertex):
            for km in kms:
                subvertex.spec.write(data=km[0])
                subvertex.spec.write(data=km[1])
                subvertex.spec.write(data=km[2])
