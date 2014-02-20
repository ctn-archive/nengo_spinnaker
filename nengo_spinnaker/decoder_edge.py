from pacman103.lib import graph

class DecoderEdge( graph.Edge ):
    def __init__(self, conn, pre, post, constraints=None, label=None):
        super(DecoderEdge, self).__init__(pre, post, constraints=constraints, 
                            label=label)
        self.index = conn.index                    

