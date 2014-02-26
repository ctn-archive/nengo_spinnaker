from pacman103.lib import graph
from pacman103.lib import lib_dsg
from pacman103.lib import lib_map
from pacman103.front.common import enums
import os
from pacman103.lib import parameters
import struct
import numpy as np

REGIONS = enums.enum1(
    'SYSTEM',
    'BIAS',
    'ENCODERS',
    'DECODERS',
    'DECODER_KEYS'
    )


class EnsembleVertex( graph.Vertex ):
    def __init__(self, data, constraints=None):
        super(EnsembleVertex, self).__init__(data.N, constraints=constraints, 
                            label=data.label)
        self.data = data                    
    
    def model_name(self):
        return 'nengo_ensemble'
    
    def get_requirements_per_atom(self):
        chip_memory = 4 + self.data.D_in*4 + self.data.D_out*4
        data_memory = chip_memory
        cycles = 1
    
        return lib_map.Resources(cycles, data_memory, chip_memory)
    
    
    def generateDataSpec(self, processor, subvertex, dao):
        IDENTIFIER = 0xABCD
        print 'generate', self.data.label
        
        spec = lib_dsg.DataSpec(processor, dao)
        spec.initialise(IDENTIFIER, dao)
        spec.comment('NEF Ensemble vertex information')
        
        x, y, p = processor.get_coordinates()
        executableTarget = lib_map.ExecutableTarget(
            dao.get_binaries_directory() + os.sep
            + 'nengo_ensemble.aplx', x, y, p)
            
        
        N = subvertex.n_neurons
            
        # size is measured in bytes    
        spec.reserveMemRegion(REGIONS.SYSTEM, size=7*4)   
        spec.reserveMemRegion(REGIONS.BIAS, size=N*4)  
        spec.reserveMemRegion(REGIONS.ENCODERS, size=N*self.data.D_in*4)  
        spec.reserveMemRegion(REGIONS.DECODERS, size=N*self.data.D_out*4)  
        spec.reserveMemRegion(REGIONS.DECODER_KEYS, size=self.data.D_out*4)  
        
        
        #TODO: adjust time resolution of sim
        #TODO: adjust realtime rate
        dt = 0.001
        spec.switchWriteFocus(REGIONS.SYSTEM)
        spec.write(data=self.data.D_in)
        spec.write(data=self.data.D_out)
        spec.write(data=N)
        spec.write(data=int(dt*(10**6)))   # dt in us
        spec.write(data=int(round(self.data.tau_ref/0.001)))   # tau_ref in steps   
        
        
        def uint(x):
            return x
        
        # 1/tau_rc in 1/seconds
        spec.write(data=uint(parameters.S1615(1. / self.data.tau_rc).converted))
        # filter decay constant
        # TODO: handle multiple filters
        
        filter = 0.01 if len(self.data.filters)==0 else self.data.filters[0]
        
        decay = np.exp(-dt/filter)
        spec.write(data=uint(parameters.S1615(decay).converted))
                
        spec.switchWriteFocus(REGIONS.BIAS)
        for i in range(N):
            spec.write(data=uint(parameters.S1615(self.data.bias[i]).converted))
        
        spec.switchWriteFocus(REGIONS.ENCODERS)
        for i in range(N):
            for j in range(self.data.D_in):
                spec.write(data=uint(parameters.S1615( 
                                    self.data.encoders[i,j] * self.data.gain[i]
                                    ).converted))
            
        decoders = self.data.get_merged_decoders()    
        spec.switchWriteFocus(REGIONS.DECODERS)
        for i in range(N):
            for j in range(self.data.D_out):
                spec.write(data=uint(parameters.S1615(decoders[i,j]/dt).converted))
            
        
        spec.switchWriteFocus(REGIONS.DECODER_KEYS)        
        index = 0
        for d,t,f in self.data.decoder_list:
            for i in range(d.shape[1]):
                spec.write(data=(x << 24) | (y << 16) | ((p-1) << 11) | (index<<6) | (i))
            index += 1    
                    
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
        
        key = (x << 24) | (y << 16) | ((p-1) << 11) | (subedge.edge.index << 6)
        
        mask = 0xFFFFFFE0
        
        return key, mask
    
        


