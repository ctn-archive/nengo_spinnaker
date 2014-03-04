from pacman103.core import dao
from pacman103.core import control
from pacman103 import conf
import sys
import numpy as np

from . import ensemble_vertex   
from . import transmit_vertex
from . import receive_vertex
from . import decoder_edge        
from . import input_edge        

from . import builder

import numpy as np
import time as pytime
import socket
import struct

class NodeRunner:
    def __init__(self, node_data, dt=0.001):
        self.node = node_data.node
        self.filter = node_data.filters[0]
        self.decay = np.exp(-dt/self.filter)
        self.value = np.zeros(self.node.size_in)
    def handle_input(self, t, key, value):
        self.value[key]= self.value[key]*decay + value*(1-decay)
        if key == self.node.size_in - 1:
            self.node.output(t, self.value)
            
            
class Simulator:
    def __init__(self, model, dt=0.001, seed=None):
        self.builder = builder.Builder(model, dt=dt, seed=seed)
                
        self.dao = dao.DAO(__name__)
        self.dao.writeTextSpecs = True
        self.make_pacman_vertices()
        self.pacman_place_and_route()
        
        
    def run(self, time):
        port = 17899
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', port))
        start = pytime.time()
        t = 0
        while t<time:
            data, addr = s.recvfrom(512)
            t = pytime.time() - start
            key, value = struct.unpack('<Ii', data[14:22])            
            value = value/(65536.0)
            print t, key, value
            
    
        
    def make_pacman_vertices(self):
    
        # make vertices for each Ensemble
        for ens in self.builder.ensembles.values():
            ens.vertex = ensemble_vertex.EnsembleVertex(ens)
            self.dao.add_vertex(ens.vertex)
        
        self.node_runners = []
        for node in self.builder.nodes.values():
            self.node_runners.append(NodeRunner(node))
            
        
        # make a Tx
        self.tx_vertex = transmit_vertex.TransmitVertex()
        self.dao.add_vertex(self.tx_vertex)
        
        # make a Rx
        self.rx_vertex = receive_vertex.ReceiveVertex()
        self.dao.add_vertex(self.rx_vertex)
        
        
        # create edges
        for c in self.builder.conn_e2e:
            self.dao.add_edge(decoder_edge.DecoderEdge(c, c.pre.vertex, c.post.vertex))
        for c in self.builder.conn_e2n:
            self.dao.add_edge(decoder_edge.DecoderEdge(c, c.pre.vertex, self.tx_vertex))
        for c in self.builder.conn_n2e:
            """If the Node is a constant, then don't bother adding this edge.
            Instead, add the value of the Node to the constant_input of the
            receiving ensemble."""
            if ( c.pre.node.output is not None
                and not callable( c.pre.node.output ) ):
                c.post.constant_input += np.asarray( c.pre.node.output )
            else:
                self.dao.add_edge(
                    input_edge.InputEdge(self.rx_vertex, c.post.vertex)
                )
            
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

        
