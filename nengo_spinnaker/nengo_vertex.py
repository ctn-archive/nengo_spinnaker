import os

from pacman103.lib import data_spec_gen, graph, lib_map
from pacman103.front.common import enums
from . import collections


class NengoVertex(graph.Vertex):
    """Generic PACMAN Vertex for custom building Nengo Nodes to override
    """

    REGIONS = enums.enum1(
        'SYSTEM'
    )

    model_name = None
    model_description = None

    def __init__(self, time_step=1000, constraints=None, label=None):
        if self.model_name is None:
            raise NotImplementedError('Must define model_name')
        if self.model_description is None:
            raise NotImplementedError('Must define model_description')
        super(NengoVertex, self).__init__(
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

    def generateDataSpec(self, processor, subvertex, dao):
        # Get the executable
        x, y, p = processor.get_coordinates()
        executable_target = lib_map.ExecutableTarget(
            os.path.join(dao.get_common_binaries_directory(),
                         '%s.aplx' % self.model_name),
            x, y, p
        )

        # Generate the spec
        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(0xABCE, dao)
        spec.comment("# %s" % self.model_description)

        spec.reserveMemRegion(1, 4)
        spec.switchWriteFocus(1)

        x, y, p = processor.get_coordinates()
        key = (x << 24) | (y << 16) | ((p-1) << 11)

        spec.write(data=key)

        spec.endSpec()
        spec.closeSpecFile()

        return (executable_target, list(), list())

    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        key = (x << 24) | (y << 16) | ((p-1) << 11)

        return key, 0xFFFFFFE0
