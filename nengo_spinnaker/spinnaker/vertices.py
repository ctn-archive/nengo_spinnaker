import collections

from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.cpu_cycles_per_tick_resource import \
    CPUCyclesPerTickResource
from pacman.model.resources.dtcm_resource import DTCMResource
from pacman.model.resources.sdram_resource import SDRAMResource


NengoPlacedVertex = collections.namedtuple(
    'NengoPlacedVertex', 'x y p executable subregions timer_period')


class NengoVertex(object):
    """Helper for constructing Vertices for PACMAN."""
    executable_path = None  # Path for the executable

    def __init__(self, n_atoms, label, regions=list(), constraints=None):
        """Create a new NengoVertex object.

        Each NengoVertex object consists of a set of regions, each region
        describes a block of memory that is to be treated in various ways.

        :param int n_atoms: Number of processing atoms represented by the
                            vertex.
        :param string label: Human readable representation for the vertex.
        :param list regions: A list of memory regions for the vertex.
        :param list constraints: A list of constraints for the vertex.
        """
        self.n_atoms = n_atoms
        self.label = label
        self.regions = regions
        self.contraints = constraints

    def get_resources_used_by_atoms(self, vertex_slice, graph):
        return ResourceContainer(
            cpu=CPUCyclesPerTickResource(
                self.get_cpu_usage_for_atoms(vertex_slice)),
            dtcm=DTCMResource(self.get_dtcm_usage_for_atoms(vertex_slice)),
            sdram=SDRAMResource(self.get_sdram_usage_for_atoms(vertex_slice))
        )

    def get_subregions(self, subvertex_index, vertex_slice):
        """Return subregions for the atoms indexed in the vertex slice.

        :param subvertex_index: The sub-object ID assigned to the subvertex.
        :param vertex_slice: The slice defining the subvertex.
        """
        if not (vertex_slice.start <= vertex_slice.stop < self.n_atoms and
                0 <= vertex_slice.start < self.n_atoms):
            raise ValueError(vertex_slice)

        return [r.create_subregion(vertex_slice, subvertex_index) for r in
                self.regions]

    def get_sdram_usage_for_atoms(self, vertex_slice):
        """Get the SDRAM usage for the given slice of the vertex.

        This calculation returns the total memory (in BYTES) used in SDRAM.
        """
        return 4*sum(r.sizeof(vertex_slice) for r in self.regions)

    def get_dtcm_usage_for_atoms(self, vertex_slice):
        """Get the DTCM usage for the given slice of the vertex.

        This calculation returns the total memory (in BYTES) used in DTCM.
        """
        words = sum(r.sizeof(vertex_slice) for r in self.regions if r.in_dtcm)
        words += self.get_dtcm_usage_static(vertex_slice)

        return 4 * words

    def get_dtcm_usage_static(self, vertex_slice):
        """Get the non-region related DTCM usage for the given number of atoms.

        This calculation returns the total memory (in BYTES) used in DTCM.
        """
        return 0

    def get_cpu_usage_for_atoms(self, vertex_slice):
        """Get the CPU usage (in ticks per step) for the given vertex slice.
        """
        raise NotImplementedError
