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
        for (i, tf) in enumerate(self.connections.transforms_functions):
            x, y, p = subvertex.placement.processor.get_coordinates()
            base = (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)
            for d in range(tf.transform.shape[0]):
                spec.write(data=base | d)

    def get_routing_id_for_connection(self, connection):
        return self.connections[connection]

    def get_routing_key_for_connection(self, subvertex, connection):
        # TODO Use edges to perform this calculation
        x, y, p = subvertex.placement.processor.get_coordinates()
        i = self.get_routing_id_for_connection(connection)
        return (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

    def generate_routing_info(self, subedge):
        key = self.get_routing_key_for_connection(subedge.presubvertex,
                                                  subedge.edge.conn)
        return key, subedge.edge.mask
