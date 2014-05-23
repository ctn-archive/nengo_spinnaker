import collections
import inspect
import struct

from pacman103.lib import graph, data_spec_gen, lib_map
from pacman103.core.utilities import memory_utils
from pacman103.core.spinnman.scp import scamp

try:
    from pkg_resources import resource_filename
except ImportError:
    import os.path

    def resource_filename(module_name, filename):
        """Get the filename for a given resource."""
        mod = __import__(module_name)
        return os.path.join(os.path.dirname(mod.__file__), filename)


region_t = collections.namedtuple('Region', ['index',
                                             'write',
                                             'pre_sizeof',
                                             'sizeof',
                                             'pre_prepare',
                                             'post_prepare'])


def Region(index, pre_sizeof, write=None, sizeof=None, pre_prepare=None,
           post_prepare=None):
    """Create a new Region instance.

    :param index: unique index of the region, will need to be mapped in C
    :param pre_sizeof: an int, or function, which represents the size of the
                       region IN WORDS (used prior to partitioning)
    :param write: a function which writes the data spec for the region
    :param sizeof: an int, or function, which represents the size of the
                   region IN WORDS (used prior to partitioning)
    :param pre_prepare: a function called prior to partitioning to prepare
                        the region
    :param post_prepare: a function called after partitioning to prepare
                         the region
    """
    return region_t(index, write, pre_sizeof, sizeof, pre_prepare,
                    post_prepare)


def ordered_regions(*args, **kwargs):
    # Assign ordered regions, then merge in manually specified regions
    regions = dict([r for r in zip(args, range(1, len(args)+1))])
    regions.update(**kwargs)

    # Ensure there are no duplicate keys
    seen = list()
    for v in regions.values():
        assert(v not in seen)
        seen.append(v)

    return regions


class NengoVertex(graph.Vertex):
    def __new__(cls, *args, **kwargs):
        """Generate the region mapping for the new instance of cls."""
        # Get a new instance, then map in the region functions
        inst = super(NengoVertex, cls).__new__(cls, *args, **kwargs)

        # Generate the region mapping for each region in turn
        inst._regions = list()
        fs = filter(lambda (_, m): hasattr(m, '_region') and
                    hasattr(m, '_region_role'), inspect.getmembers(inst))

        for (region, index) in cls.REGIONS.items():
            r_fs = filter(lambda (_, m): m._region == region, fs)
            mapped = dict([(m._region_role, m) for (_, m) in r_fs])
            assert("pre_sizeof" in mapped)
            inst._regions.append(Region(index, **mapped))

        inst.runtime = None

        return inst

    @property
    def model_name(self):
        return self.MODEL_NAME

    def pre_prepare(self):
        """Prepare vertex, called prior to partitioning."""
        for region in self._regions:
            if region.pre_prepare is not None:
                region.pre_prepare()

    def post_prepare(self):
        """Prepare vertex, called prior to partitioning."""
        for region in self._regions:
            if region.post_prepare is not None:
                region.post_prepare()

    def __pre_sizeof_regions(self, n_atoms):
        return sum([r.pre_sizeof(n_atoms) for r in self._regions]) * 4

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                *args):
        n_atoms = hi_atom - lo_atom + 1

        sdram_usage = self.__pre_sizeof_regions(n_atoms)
        dtcm_usage = sdram_usage

        if hasattr(self, 'dtcm_usage'):
            if callable(self.dtcm_usage):
                dtcm_usage = self.dtcm_usage(n_atoms)
            else:
                dtcm_usage = self.dtcm_usage

        return lib_map.Resources(self.cpu_usage(n_atoms), dtcm_usage,
                                 sdram_usage)

    def generateDataSpec(self, processor, subvertex, dao):
        # Create a spec, reserve regions and fill in as necessary
        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(0xABCD, dao)
        self.__reserve_regions(subvertex, spec)
        self.__write_regions(subvertex, spec)
        spec.endSpec()
        spec.closeSpecFile()

        # Write the runtime to the core
        x, y, p = processor.get_coordinates()
        run_ticks = ((1 << 32) - 1 if self.runtime is None else
                     self.runtime * 1000)  # TODO Deal with timestep scaling

        addr = 0xe5007000 + 128 * p + 116  # Space reserved for _p_
        mem_writes = [lib_map.MemWriteTarget(x, y, p, addr, run_ticks)]

        # Get the executable
        executable_target = lib_map.ExecutableTarget(
            resource_filename("nengo_spinnaker",
                              "binaries/%s.aplx" % self.MODEL_NAME),
            x, y, p
        )

        return (executable_target, list(), mem_writes)

    def __reserve_regions(self, subvertex, spec):
        for region in self._regions:
            size = (region.pre_sizeof(subvertex.n_atoms) if region.sizeof is
                    None else region.sizeof(subvertex))
            unfilled = region.write is None

            if size > 0:
                spec.reserveMemRegion(region.index, size*4,
                                      leaveUnfilled=unfilled)

    def __write_regions(self, subvertex, spec):
        for region in self._regions:
            size = (region.pre_sizeof(subvertex.n_atoms) if region.sizeof is
                    None else region.sizeof(subvertex))
            if region.write is not None and size > 0:
                spec.switchWriteFocus(region.index)
                region.write(subvertex, spec)


def _region_role_mark(region, role):
    def f_(f):
        f._region = region
        f._region_role = role
        return f
    return f_


def region_pre_sizeof(region):
    return _region_role_mark(region, "pre_sizeof")


def region_sizeof(region):
    return _region_role_mark(region, "sizeof")


def region_write(region):
    return _region_role_mark(region, "write")


def region_pre_prepare(region):
    return _region_role_mark(region, "pre_prepare")


def region_post_prepare(region):
    return _region_role_mark(region, "post_prepare")


def retrieve_region_data(txrx, x, y, p, region_id, region_size):
    """Get the data from the given processor and region.

    :param txrx: transceiver to use when communicating with the board
    :param region_id: id of the region to retrieve
    :param region_size: size of the region (in words)
    :returns: a string containing data from the region
    """
    # Get the application pointer table to get the address for the region
    txrx.select(x, y)
    app_data_base_offset = memory_utils.getAppDataBaseAddressOffset(p)
    _app_data_table = txrx.memory_calls.read_mem(app_data_base_offset,
                                                 scamp.TYPE_WORD, 4)
    app_data_table = struct.unpack('<I', _app_data_table)[0]

    # Get the position of the desired region
    region_base_offset = memory_utils.getRegionBaseAddressOffset(
        app_data_table, region_id)
    _region_base = txrx.memory_calls.read_mem(region_base_offset,
                                              scamp.TYPE_WORD, 4)
    region_address = struct.unpack('<I', _region_base)[0] + app_data_table

    # Read the region
    data = txrx.memory_calls.read_mem(region_address, scamp.TYPE_WORD,
                                      region_size * 4)
    return data
