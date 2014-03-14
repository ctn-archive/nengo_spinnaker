from pacman103.lib import graph


class InputEdge( graph.Edge ):
    def __init__(self, conn, pre, post, constraints=None, label=None):
        super(InputEdge, self).__init__(
            pre, post, constraints=constraints, label=label
        )
        self.conn = conn
