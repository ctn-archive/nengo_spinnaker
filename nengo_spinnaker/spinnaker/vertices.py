class NengoVertex(object):
    """Helper for constructing Vertices for PACMAN."""
    def __init__(self, n_atoms, label, executable, max_atoms_per_core,
                 regions=list(), constraints=None):
        """Create a new NengoVertex object.

        Each NengoVertex object consists of a set of regions, each region
        describes a block of memory that is to be treated in various ways.

        :param int n_atoms: Number of processing atoms represented by the
                            vertex.
        :param string label: Human readable representation for the vertex.
        :param string executable: Complete path of executable to use for
                                  subvertices of this vertex.
        :param int max_atoms_per_core: The maximum number of atoms to allocate
                                       to a processor core.
        :param list regions: A list of memory regions for the vertex.
        :param list constraints: A list of constraints for the vertex.
        """
        raise NotImplementedError
