import collections

from .. import utils
from ..utils import vertices


node_transform_entry_t = collections.namedtuple(
    'NodeTransformEntry', ['node', 'transform', 'width']
)


def NodeTransformEntry(node, transform, width):
    t = utils.totuple(transform)
    return node_transform_entry_t(node, t, width)


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
        self.assigned_nodes_transforms = list()

    @property
    def n_assigned_dimensions(self):
        return sum([e.width for e in self.assigned_nodes_transforms])

    @property
    def n_remaining_dimensions(self):
        return self.MAX_DIMENSIONS - self.n_assigned_dimensions

    def add_node_transform(self, node, transform, width):
        nt = NodeTransformEntry(node, transform, width)
        self.assigned_nodes_transforms.append(nt)

    def get_node_transform_offset(self, node, transform):
        """Get the offset of the given Node and transform in the vector output
        Vector of the Node.

        :raises KeyError: if the Node and transform are not assigned to this
                          Vertex.
        """
        offset = 0
        for nte in self.assigned_nodes_transforms:
            if nte.node == node and nte.transform == utils.totuple(transform):
                return offset
            offset += nte.width
        else:
            raise KeyError

    def get_maximum_atoms_per_core(self):
        return 1

    def cpu_usage(self, n_atoms):
        return 1

    @vertices.region_pre_sizeof('SYSTEM')
    def sizeof_region_system(self, n_atoms):
        return 2

    @vertices.region_pre_sizeof('OUTPUT_KEYS')
    def sizeof_region_output_keys(self, n_atoms):
        """Get the size (in bytes) of the OUTPUT_KEYS region."""
        # 1 word per edge
        return 4 * len(self.out_edges)

    @vertices.region_write('SYSTEM')
    def write_region_system(self, subvertex, spec):
        spec.write(data=1000/self.n_assigned_dimensions)
        spec.write(data=self.n_assigned_dimensions)

    @vertices.region_write('OUTPUT_KEYS')
    def write_region_output_keys(self, subvertex, spec):
        for nte in self.assigned_nodes_transforms:
            base = self.get_routing_key_for_node_transform(
                subvertex, nte.node, nte.transform
            )
            for d in range(nte.width):
                spec.write(data=base | d)

    def get_routing_id_for_node_transform(self, node, transform):
        """Get the routing ID for the given Node and transform.

        :raises KeyError: if the Node and Transform are not assigned to this
                          vertex.
        """
        for i, nte in enumerate(self.assigned_nodes_transforms):
            if nte.node == node and nte.transform == utils.totuple(transform):
                return i
        else:
            raise KeyError

    def get_routing_key_for_node_transform(self, subvertex, node, transform):
        """Get the routing key for the given subvertex, Node and transform."""
        x, y, p = subvertex.placement.processor.get_coordinates()
        i = self.get_routing_id_for_node_transform(node, transform)
        return (x << 24) | (y << 16) | ((p-1) << 11) | (i << 6)

    def generate_routing_info(self, subedge):
        key = self.get_routing_key_for_node_transform(subedge.presubvertex,
                                                      subedge.edge.pre,
                                                      subedge.edge.transform)
        return key, 0xFFFFFFE0
