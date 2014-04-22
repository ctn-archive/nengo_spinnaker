from . import receive_vertex, transmit_vertex


class Ethernet(object):
    def __init__(self, address, port=5555):
        self.address = address
        self.port = port

        self._tx_vertices = list()
        self._tx_assigns = list()
        self._rx_vertices = list()
        self._rx_assigns = list()

    def build_node(self, builder, node):
        """Build the given Node
        """
        # If the Node has input, then assign the Node to a Tx component
        if node.size_in > 0:
            # Try to fit the Node in an existing Tx Element
            # Most recently added Txes are nearer the start
            for tx in self._tx_vertices:
                if tx.remaining_dimensions >= node.size_in:
                    tx.assign_node(node)
                    self._tx_assigns[node] = tx
                    break
            else:
                # Otherwise create a new Tx element
                tx = transmit_vertex.TransmitVertex(
                    label="Tx%d" % len(self._tx_vertices)
                )
                builder.add_vertex(tx)
                tx.assign_node(node)
                self._tx_assigns[node] = tx
                self._tx_vertices.insert(0, tx)

        # If the Node has output, and that output is not constant, then assign
        # the Node to an Rx component.
        if node.size_out > 0 and callable(node.output):
            # Try to fit the Node in an existing Rx Element
            # Most recently added Rxes are nearer the start
            for rx in self._rx_vertices:
                if rx.remaining_dimensions >= node.size_out:
                    rx.assign_node(node)
                    rx_assigned = True
                    self._rx_assigns[node] = rx
                    break
            else:
                # Otherwise create a new Rx element
                rx = receive_vertex.ReceiveVertex(
                    label="Rx%d" % len(self._rx_vertices)
                )
                builder.add_vertex(rx)
                rx.assign_node(node)
                self._rx_assigns[node] = rx
                self._rx_vertices.insert(0, rx)

    def get_node_in_vertex(self, builder, c):
        """Get the Vertex which accepts input for the Node
        """
        return self._tx_assigns[c.post]

    def get_node_out_vertex(self, builder, c):
        """Get the Vertex which transmits output from the Node
        """
        return self._rx_assigns[c.pre]

    def get_node_input(self, node):
        raise NotImplementedError

    def set_node_output(self, node):
        raise NotImplementedError
