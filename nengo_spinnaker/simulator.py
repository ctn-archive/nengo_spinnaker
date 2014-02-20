from pacman103.core import dao
from pacman103.core import control
from pacman103 import conf
import sys

from . import ensemble_vertex   
from . import transmit_vertex
from . import decoder_edge        
from . import input_edge        

from . import builder


class Simulator:
    def __init__(self, model, dt=0.001, seed=None):
        self.builder = builder.Builder(model, dt=dt, seed=seed)
                
        self.dao = dao.DAO(__name__)
        self.dao.writeTextSpecs = True
        self.make_pacman_vertices()
        self.pacman_place_and_route()
        
        
    def run(self, time):
        pass    
        
    def make_pacman_vertices(self):
    
        # make vertices for each Ensemble
        for ens in self.builder.ensembles.values():
            ens.vertex = ensemble_vertex.EnsembleVertex(ens)
            self.dao.add_vertex(ens.vertex)
        
        # make a Tx
        self.tx_vertex = transmit_vertex.TransmitVertex()
        self.dao.add_vertex(self.tx_vertex)
        
        # make a Rx
        # TODO
        
        
        # create edges
        for c in self.builder.conn_e2e:
            self.dao.add_edge(decoder_edge.DecoderEdge(c, c.pre.vertex, c.post.vertex))
        for c in self.builder.conn_e2n:
            self.dao.add_edge(decoder_edge.DecoderEdge(c, c.pre.vertex, self.tx_vertex))
        for c in self.builder.conn_n2e:
            self.dao.add_edge(input_edge.InputEdge(c, self.rx_vertex, c.post.vertex))
            
    def pacman_place_and_route(self):
        controller = control.Controller(sys.modules[__name__], 
                                conf.config.get('Machine', 'machineName'))
        controller.dao = self.dao   
        self.dao.set_hostname(conf.config.get('Machine', 'machineName'))
        controller.map_model()
        controller.generate_output()  
        controller.load_targets()
        controller.load_write_mem() 
        controller.run(controller.dao.app_id)                     

        
