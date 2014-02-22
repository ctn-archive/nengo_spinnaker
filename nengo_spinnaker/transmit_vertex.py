from pacman103.lib import graph
from pacman103.lib import lib_dsg
from pacman103.lib import lib_map
from pacman103.front.common import enums
import os
from pacman103.lib import parameters


class TransmitVertex( graph.Vertex ):
    def __init__(self, constraints=None, label=None):
        super(TransmitVertex, self).__init__(1, constraints=constraints, 
                            label=label)

    def model_name(self):
        return 'nengo_tx'
    
    def get_requirements_per_atom(self):
        chip_memory = 0
        data_memory = 0
        cycles = 1
    
        return lib_map.Resources(cycles, data_memory, chip_memory)

    def generateDataSpec(self, processor, subvertex, dao):
        IDENTIFIER = 0xABCE
        
        x, y, p = processor.get_coordinates()
        executableTarget = lib_map.ExecutableTarget(
            dao.get_binaries_directory() + os.sep
            + 'nengo-tx.aplx', x, y, p)


        spec = lib_dsg.DataSpec(processor, dao)
        spec.initialise(IDENTIFIER, dao)
        spec.comment('Nengo transmitter')
        # End the writing of this specification:
        spec.endSpec()
        spec.closeSpecFile() 

            
        # No memory writes required for this Data Spec:
        memoryWriteTargets = list()
        loadTargets = list()

        # Return list of target cores, executables, files to load and 
        # memory writes to perform:
        return  executableTarget, loadTargets, memoryWriteTargets
        
        
        
    
                            
    def generate_routing_info(self, subedge):
        x, y, p = subedge.presubvertex.placement.processor.get_coordinates()
        
        key = (x << 24) | (y << 16) | ((p-1) << 11)
        
        mask = 0xFFFFFFE0
        
        return key, mask
    

