from pacman103.lib import graph

class DecoderEdge( graph.Edge ):
    def __init__(self, conn, pre, post, constraints=None, label=None):
        super(DecoderEdge, self).__init__(pre, post, constraints=constraints, 
                            label=label)
        self.index = conn.index                    
        self.conn = conn
        self.key = None
    def add_tx_key(self, key):
        assert self.key is None
        #TODO: what happens if this edge is split into sub edges?
        self.key = key
