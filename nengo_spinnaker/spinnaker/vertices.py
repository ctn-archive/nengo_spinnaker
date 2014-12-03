import collections


PlacedVertex = collections.namedtuple(
    'PlacedVertex', 'x y p executable subregions timer_period')


class Vertex(object):
    """Represents an instance (or set of instances) of an executable on a
    SpiNNaker machine.
    """
    executable_path = None  # Path for the executable

    def __init__(self, n_atoms, label, regions=list(), constraints=None):
        """Create a new Vertex object.

        Each Vertex object consists of a set of regions, each region
        describes a block of memory that is to be treated in various ways.

        Parameters
        ----------
        n_atoms : int
            Number of processing atoms represented by the vertex.
        label : string
            Human readable representation for the vertex.
        regions : list
            A list of memory regions for the vertex.
        constraints : list
            A list of constraints for the vertex.
        """
        self.label = label
        self.constrains = constraints if constraints is not None else list()
        self.n_atoms = n_atoms
        self.regions = regions

    def get_subregions(self, subvertex_index, vertex_slice):
        """Return subregions for the atoms indexed in the vertex slice.

        Parameters
        ----------
        subvertex_index : int
            Index of the subvertex for which subregions are desired.
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms for which subregions should be generated.

        Returns
        -------
        list :
            A list of :py:func:`Subregion`s.
        """
        if vertex_slice.stop > self.n_atoms:
            raise ValueError(
                "Attempt to retrieve data for more atoms than present.")
        return [r.create_subregion(vertex_slice, subvertex_index) for r in
                self.regions]

    def get_sdram_usage_for_atoms(self, vertex_slice):
        """Get the SDRAM usage for the given slice of the vertex.

        This calculation returns the total memory (in BYTES) used in SDRAM.
        Method is not intended to be overridden.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms to get the usage requirements of.
        """
        return 4*sum(r.sizeof(vertex_slice) for r in self.regions if
                     r is not None)

    def get_dtcm_usage_for_atoms(self, vertex_slice):
        """Get the DTCM usage for the given slice of the vertex.

        This calculation returns the total memory (in BYTES) used in DTCM.
        Method is not intended to be overridden.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms to get the usage requirements of.
        """
        words = sum(r.sizeof(vertex_slice) for r in self.regions if
                    (r is not None and r.in_dtcm))
        words += self.get_dtcm_usage_static(vertex_slice)

        return 4 * words

    def get_dtcm_usage_static(self, vertex_slice):
        """Get the non-region related DTCM usage for the given number of atoms.

        This calculation returns the total memory (in BYTES) used in DTCM.
        Method is intended to be overridden.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms to get the usage requirements of.
        """
        return 0

    def get_cpu_usage_for_atoms(self, vertex_slice):
        """Get the CPU usage (in ticks per step) for the given vertex slice.

        Method is intended to be overridden.

        Parameters
        ----------
        vertex_slice : :py:class:`nengo_vertex.spinnaker.partitioners.Slice`
            The slice of atoms to get the usage requirements of.
        """
        raise NotImplementedError
