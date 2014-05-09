try:
    from pkg_resources import resource_filename
except ImportError:
    import os.path

    def resource_filename(module_name, filename):
        """Get the filename for a given resource."""
        mod = __import__(module_name)
        return os.path.join(os.path.dirname(mod.__file__), filename)

from .utils import fp


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
