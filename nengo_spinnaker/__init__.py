# from .simulator import Simulator  # noqa

# Register transform functions with the builder
from builder import Builder
import ensemble.build
from ensemble import pes
# import probe

Builder.register_object_transform(ensemble.build.build_ensembles)
# Builder.register_connectivity_transform(
#     probe.insert_decoded_output_probes)
Builder.register_connectivity_transform(pes.reroute_modulatory_connections)

# Register assembler methods with the assembler
import assembler
import connection
# import ensemble.lif

assembler.Assembler.register_connection_builder(
    connection.generic_connection_builder)
# assembler.Assembler.register_object_builder(
#     ensemble.EnsembleLIF.assemble, ensemble.IntermediateEnsembleLIF)
# assembler.Assembler.register_object_builder(
#     node.FilterVertex.assemble_from_intermediate, node.IntermediateFilter)
# assembler.Assembler.register_object_builder(
#     node.FilterVertex.assemble, node.FilterVertex)
# assembler.Assembler.register_object_builder(
#     probe.DecodedValueProbe.assemble, probe.IntermediateProbe)
