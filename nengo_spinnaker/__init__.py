# from .simulator import Simulator  # noqa

# Register transform functions with the builder
import ensemble.build  # noqa
# import probe

# Builder.register_object_transform(ensemble.build.build_ensembles)
# Builder.register_connectivity_transform(
#     probe.insert_decoded_output_probes)

# Register assembler methods with the assembler
from .assembler import Assembler
import connection
from .ensemble import lif

Assembler.register_connection_builder(connection.generic_connection_builder)
Assembler.register_object_builder(lif.EnsembleLIF.assemble_from_intermediate,
                                  lif.IntermediateLIF)
# assembler.Assembler.register_object_builder(
#     node.FilterVertex.assemble_from_intermediate, node.IntermediateFilter)
# assembler.Assembler.register_object_builder(
#     node.FilterVertex.assemble, node.FilterVertex)
# assembler.Assembler.register_object_builder(
#     probe.DecodedValueProbe.assemble, probe.IntermediateProbe)
