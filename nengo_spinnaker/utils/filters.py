import collections
import numpy as np

from . import fixpoint as fp
from . import connections
from vertices import (region_pre_sizeof, region_sizeof, region_write,
                      region_pre_prepare, region_post_prepare)


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
    return 3 * self.n_filters + 1


@region_pre_sizeof("FILTER_ROUTING")
def _pre_sizeof_region_filter_routing(self, n_atoms):
    return 4 * len(self.in_edges) * 5


@region_pre_prepare('FILTERS')
def _pre_prepare_filters(self):
    """Generate a list of filters from the incoming edges."""
    self.__filters = connections.Filters(
        [connections.ConnectionWithFilter(edge.conn,
                                          edge._filter_is_accumulatory) for
         edge in self.in_edges]
    )
    self.n_filters = len(self.__filters)


@region_write("FILTERS")
def _write_region_filters(self, subvertex, spec):
    spec.write(data=len(self.__filters))
    for filter_item in self.__filters.filters:
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
    self.__filter_keys = list()

    for edge in self.in_edges:
        i = self.__filters[edge.conn]
        dmask = edge.dimension_mask

        routings = [edge.prevertex.generate_routing_info(se) for se in
                    edge.subedges]

        self.__filter_keys.extend(
            [FilterRoute(r[0], r[1], i, dmask) for r in routings])


@region_sizeof("FILTER_ROUTING")
def _sizeof_region_filter_routing(self, subvertex):
    # 4 words per entry, 1 entry per in_subedge + 1 for length
    return 4 * len(self.__filter_keys) + 1


@region_write("FILTER_ROUTING")
def _write_region_filter_routing(self, subvertex, spec):
    spec.write(data=len(self.__filter_keys))
    for route in self.__filter_keys:
        spec.write(data=route.key)
        spec.write(data=route.mask)
        spec.write(data=route.index)
        spec.write(data=route.dimension_mask)
