from .. import utils
from ..utils import connections, vertices


class SDPReceiveVertex(vertices.NengoVertex):
    """PACMAN Vertex for an object which receives input from Nodes on the host
    and forwards it to connected Ensembles.
    """
    REGIONS = vertices.ordered_regions('SYSTEM', 'OUTPUT_KEYS')
    MAX_DIMENSIONS = 64
    MODEL_NAME = "nengo_rx"

    def __init__(self, time_step=1000, constraints=None, label=None):
        super(SDPReceiveVertex, self).__init__(1, constraints=constraints,
                                               label=label)
        self.connections = connections.Connections()

    @property
    def n_assigned_dimensions(self):
        return self.connections.width

    @property
    def n_remaining_dimensions(self):
        return self.MAX_DIMENSIONS - self.n_assigned_dimensions

    def add_connection(self, connection):
        self.connections.add_connection(connection)

    def get_connection_offset(self, connection):
        return self.connections.get_connection_offset(connection)

    def contains_compatible_connection(self, connection):
        return self.connections.contains_compatible_connection(connection)

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        return 2

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_region_output_keys(self, n_atoms):
        """Get the size (in words) of the OUTPUT_KEYS region."""
        return self.connections.width

    @vertices.region_write('SYSTEM')
    def write_region_system(self, subvertex, spec):
        spec.write(data=1000)
        spec.write(data=self.n_assigned_dimensions)

    @vertices.region_write('OUTPUT_KEYS')
    def write_region_output_keys(self, subvertex, spec):
        # Ensure the keys generated for edges match what we're about to write
        self.out_connections = self.connections

        # For each unique output connection write the key we'll be using
        for (i, tf) in enumerate(self.connections.transforms_functions):
            x, y, p = subvertex.placement.processor.get_coordinates()
            for d in range(tf.transform.shape[0]):
                if not tf.keyspace.is_set_i:
                    spec.write(data=tf.keyspace.key(x=x, y=y, p=p-1, i=i, d=d))
                else:
                    spec.write(data=tf.keyspace.key(x=x, y=y, p=p-1, d=d))
