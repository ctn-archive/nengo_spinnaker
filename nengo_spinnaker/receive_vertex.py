import os

from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums
from . import collections


class ReceiveVertex(graph.Vertex):
    """PACMAN Vertex for an object which receives input from Nodes on the host
    and forwards it to connected Ensembles.
    """

    REGIONS = enums.enum1(
        'SYSTEM'
    )
    MAX_DIMENSIONS = 64

    model_name = "nengo_rx"

    def __init__(self, time_step=1000, constraints=None, label=None):
        # Dimension management
        self._assigned_dimensions = 0
        self._assigned_nodes = collections.AssignedNodeBin(
            self.MAX_DIMENSIONS, lambda n: n.size_out
        )

        # Create the vertex
        super(ReceiveVertex, self).__init__(
            1, constraints=constraints, label=label
        )

    def get_maximum_atoms_per_core(self):
        return 1

    def get_resources_for_atoms(self, lo_atom, hi_atom, n_machine_time_steps,
                                machine_time_step_us, partition_data_object):
        return lib_map.Resources(1, 1, 1)

    @property
    def remaining_dimensions(self):
        return self._assigned_nodes.remaining_space

    def assign_node(self, node):
        """Assign a Nengo Node to this ReceiveVertex."""
        self._assigned_nodes.append(node)

    @property
    def nodes(self):
        """Return the Nodes assigned to this ReceiveVertex."""
        return self._assigned_nodes.nodes

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(dao.get_common_binaries_directory(), 'nengo_rx.aplx'),
            x, y, p
        )

        # Generate the spec
        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(0xABCE, dao)
        spec.comment("# Nengo Rx Component")

        spec.endSpec()
        spec.closeSpecFile()

        return (executable_target, list(), list())

    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        key = (x << 24) | (y << 16) | ((p-1) << 11)

        return key
