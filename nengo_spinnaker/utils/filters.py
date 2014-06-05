import collections
import itertools
import numpy as np

from . import fixpoint as fp
from vertices import (region_pre_sizeof, region_sizeof, region_write,
                      region_pre_prepare, region_post_prepare)


FilterItem = collections.namedtuple('FilterItem', ['time_constant',
                                                   'accumulatory'])


FilterRoute = collections.namedtuple('FilterRoute', ['key', 'mask', 'index',
                                                     'dimension_mask'])


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

        cls._prep_region_filters = _pre_prepare_filters
        cls._prep_region_filter_routing = _post_prepare_routing

        return cls
    return cls_


@region_pre_sizeof("FILTERS")
def _sizeof_region_filters(self, n_atoms):
    # 3 words per filter + 1 for length
    return 3 * len(self.__filters) + 1


@region_pre_sizeof("FILTER_ROUTING")
def _pre_sizeof_region_filter_routing(self, n_atoms):
    return 4 * len(self.in_edges) * 5


@region_pre_prepare('FILTERS')
def _pre_prepare_filters(self):
    """Generate a list of filters from the incoming edges."""
    self.__filters = list()
    self.__filters_in = collections.defaultdict(list)

    for edge in self.in_edges:
        filter_item = FilterItem(edge.synapse, edge._filter_is_accumulatory)

        if filter_item not in self.__filters:
            self.__filters.append(filter_item)

        self.__filters_in[self.__filters.index(filter_item)].append(edge)

    self.n_filters = len(self.__filters)


@region_write("FILTERS")
def _write_region_filters(self, subvertex, spec):
    spec.write(data=len(self.__filters))
    for filter_item in self.__filters:
        f = (np.exp(-self.dt / filter_item.time_constant) if
             filter_item.time_constant is not None else 0.)
        spec.write(data=fp.bitsk(f))
        spec.write(data=fp.bitsk(1 - f))
        spec.write(data=(0x0 if filter_item.accumulatory else 0xffffffff))


@region_post_prepare('FILTER_ROUTING')
def _post_prepare_routing(self):
    # For each incoming subedge we write the key, mask and index of the
    # filter to which it is connected.  At some later point we can try
    # to combine keys and masks to minimise the number of comparisons
    # which are made in the SpiNNaker application.

    # Mapping of subvertices to list of maps from keys and masks to filter
    # indices (filter routing entries)
    self.__subvertex_filter_keys = collections.defaultdict(list)

    for (i, edges) in self.__filters_in.items():
        for subvertex in self.subvertices:
            subedges = itertools.chain(*[
                filter(lambda se: se.postsubvertex == subvertex,
                       edge.subedges) for edge in edges]
            )
            kms = [(subedge.edge.prevertex.generate_routing_info(subedge),
                    subedge.edge.dimension_mask) for subedge in subedges]

            # Add the key and mask entries to the filter keys list for this
            # subvertex.
            self.__subvertex_filter_keys[subvertex].extend(
                [FilterRoute(km[0], km[1], i, dm) for (km, dm) in kms]
            )


@region_sizeof("FILTER_ROUTING")
def _sizeof_region_filter_routing(self, subvertex):
    # 4 words per entry, 1 entry per in_subedge + 1 for length
    return 4 * len(self.__subvertex_filter_keys[subvertex]) + 1


@region_write("FILTER_ROUTING")
def _write_region_filter_routing(self, subvertex, spec):
    routes = self.__subvertex_filter_keys[subvertex]

    spec.write(data=len(routes))
    for route in routes:
        spec.write(data=route.key)
        spec.write(data=route.mask)
        spec.write(data=route.index)
        spec.write(data=route.dimension_mask)
