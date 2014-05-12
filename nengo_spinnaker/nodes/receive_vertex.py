import collections
import os

from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums

from .. import utils
from .. import vertices


node_transform_entry_t = collections.namedtuple(
    'NodeTransformEntry', ['node', 'transform', 'width']
)


def NodeTransformEntry(node, transform, width):
    t = utils.totuple(transform)
    return node_transform_entry_t(node, t, width)


class ReceiveVertex(graph.Vertex):
    """PACMAN Vertex for an object which receives input from Nodes on the host
    and forwards it to connected Ensembles.
    """

    REGIONS = enums.enum1(
        'SYSTEM',
        'OUTPUT_KEYS'
    )
    MAX_DIMENSIONS = 64

    model_name = "nengo_rx"

    def __init__(self, time_step=1000, constraints=None, label=None):
        super(ReceiveVertex, self).__init__(1, constraints=constraints,
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

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                machine_time_step_us, partition_data_object):
        return lib_map.Resources(1, 1, 1)

    def sizeof_region_system(self):
        """Get the size (in bytes) of the SYSTEM region."""
        # 2 words
        return 4 * 2

    def sizeof_region_output_keys(self):
        """Get the size (in bytes) of the OUTPUT_KEYS region."""
        # 1 word per edge
        return 4 * len(self.out_edges)

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            vertices.resource_filename("nengo_spinnaker",
                                       "binaries/%s.aplx" % self.model_name),
            x, y, p
        )

        # Generate the spec
        subvertex.spec = data_spec_gen.DataSpec(processor, dao)
        subvertex.spec.initialise(0xABCE, dao)
        subvertex.spec.comment("# Nengo Rx Component")

        # Fill in the spec
        self.reserve_regions(subvertex)
        self.write_region_system(subvertex)
        self.write_region_output_keys(subvertex)

        subvertex.spec.endSpec()
        subvertex.spec.closeSpecFile()

        return (executable_target, list(), list())

    def reserve_regions(self, subvertex):
        subvertex.spec.reserveMemRegion(
            self.REGIONS.SYSTEM,
            self.sizeof_region_system()
        )
        subvertex.spec.reserveMemRegion(
            self.REGIONS.OUTPUT_KEYS,
            self.sizeof_region_output_keys()
        )

    def write_region_system(self, subvertex):
        subvertex.spec.switchWriteFocus(self.REGIONS.SYSTEM)
        subvertex.spec.comment("""# System Region
        # -------------
        # 1. Number of us between transmitting MC packets
        # 2. Number of dimensions
        """)
        subvertex.spec.write(data=1000/self.n_assigned_dimensions)
        subvertex.spec.write(data=self.n_assigned_dimensions)

    def write_region_output_keys(self, subvertex):
        subvertex.spec.switchWriteFocus(self.REGIONS.OUTPUT_KEYS)
        subvertex.spec.comment("# Output Keys")

        for nte in self.assigned_nodes_transforms:
            base = self.get_routing_key_for_node_transform(
                subvertex, nte.node, nte.transform
            )
            for d in range(nte.width):
                subvertex.spec.write(data=base | d)

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
