import collections
import inspect

from pacman103.lib import graph, data_spec_gen, lib_map

import fixpoint as fp

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


def Region(index, write, pre_sizeof, sizeof=None, pre_prepare=None,
           post_prepare=None):
    """Create a new Region instance.

    :param index: unique index of the region, will need to be mapped in C
    :param write: a function which writes the data spec for the region
    :param pre_sizeof: an int, or function, which represents the size of the
                       region IN WORDS (used prior to partitioning)
    :param sizeof: an int, or function, which represents the size of the
                   region IN WORDS (used prior to partitioning)
    :param pre_prepare: a function called prior to partitioning to prepare
                        the region
    :param post_prepare: a function called after partitioning to prepare
                         the region
    """
    return region_t(index, write, pre_sizeof, sizeof, pre_prepare,
                    post_prepare)


def ordered_regions(*args):
    return dict([r for r in zip(args, range(1, len(args)+1))])


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
            assert("write" in mapped and "pre_sizeof" in mapped)
            inst._regions.append(Region(index, **mapped))

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

        dtcm_usage = self.__pre_sizeof_regions(n_atoms)
        sdram_usage = dtcm_usage  # TODO Break this assumption when necessary

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

        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            resource_filename("nengo_spinnaker",
                              "binaries/%s.aplx" % self.MODEL_NAME),
            x, y, p
        )

        return (executable_target, list(), list())

    def __reserve_regions(self, subvertex, spec):
        for region in self._regions:
            size = (region.pre_sizeof(subvertex.n_atoms) if region.sizeof is
                    None else region.sizeof(subvertex))
            spec.reserveMemRegion(region.index, size * 4)

    def __write_regions(self, subvertex, spec):
        for region in self._regions:
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
