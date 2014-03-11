from pacman103.lib import graph
from pacman103.lib import data_spec_gen
from pacman103.lib import lib_map
from pacman103.front.common import enums
import os
from pacman103.lib import parameters

REGIONS = enums.enum1(
    'SYSTEM',
    'KEYS',
    'INITIAL_VALUES'
)

class ReceiveVertex( graph.Vertex ):
    def __init__(self, constraints=None, label=None):
        super(ReceiveVertex, self).__init__(1, constraints=constraints, 
                            label=label)

    def model_name(self):
        return 'nengo_rx'
    
    def get_requirements_per_atom(self):
        chip_memory = 0
        data_memory = 0
        cycles = 1
    
        return lib_map.Resources(cycles, data_memory, chip_memory)

    @property
    def n_dims( self ):
        """The number of dimensions this Rx component will represent.
        Must be less than 256/4 = 64.  This is the sum of the number of input
        dimensions for all receiving Ensembles.

        TODO: Rename this attribute or do it differently somehow..?"""
        return sum( map( lambda e : e.postvertex.data.D_in, self.out_edges ) )

    def generateDataSpec(self, processor, subvertex, dao):
        IDENTIFIER = 0xABCE
        
        x, y, p = processor.get_coordinates()
        executableTarget = lib_map.ExecutableTarget(
            dao.get_binaries_directory() + os.sep
            + 'nengo_rx.aplx', x, y, p)
 
        spec = data_spec_gen.DataSpec(processor, dao)
        spec.initialise(IDENTIFIER, dao)
        spec.comment('Nengo receiver')
        spec.reserveMemRegion( REGIONS.SYSTEM, size = 2 * 4 )
        spec.reserveMemRegion( REGIONS.KEYS, size = self.n_dims * 4 )
        spec.reserveMemRegion( REGIONS.INITIAL_VALUES, size = self.n_dims * 4 )

        spec.comment(
            """
            # System Parameters
            # 1. Number of dimensions to represent.
            # 2. Interval between transmitting MC packets
            """
        )
        spec.switchWriteFocus(REGIONS.SYSTEM)
        spec.write( data = self.n_dims )
        # TODO: adjust time resolution of sim
        spec.write(data=int(0.001 * 10**6)/self.n_dims)

        spec.comment(
        """Dimension related routing keys"""
        )
        spec.switchWriteFocus(REGIONS.KEYS)
        for e in self.out_edges:
            """
            We need a routing key for each output edge and dimension.
            This needs to be in the same format as that for the EnsembleVertex.
            """
            for d in range( e.postvertex.data.D_in ):
                key = self.generate_routing_info( e.subedges[0] )[0] | (d)
                spec.write( data = key )

        spec.comment(
            """Initial values for dimensions, defaulting to 0."""
        )
        spec.switchWriteFocus(REGIONS.INITIAL_VALUES)
        for d in range( self.n_dims ):
            spec.write( data = 0x00000000 )

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

        # Get the index of this edge in the list of subedges we have
        i = self.out_edges.index( subedge.edge )
        
        key = (x << 24) | (y << 16) | ((p-1) << 11) | ( i << 6 )
        
        mask = 0xFFFFFFE0
        
        return key, mask
    

